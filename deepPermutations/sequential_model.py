import collections
import os
import pickle
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F
from deepPermutations.data_preprocessing import SOP_INDEX, indexed_seq_to_score
from deepPermutations.data_utils import PACKAGE_DIR, \
    variable2numpy, SEQ, numpy2variable
from deepPermutations.data_utils import START_SYMBOL, END_SYMBOL, \
    first_note_index
from deepPermutations.permutation_distance import spearman_rho
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm


class SequentialModel(nn.Module):
    """
    Base class for distance models
    """

    def __init__(self, model_type: str, dataset_name: str,
                 num_pitches=None,
                 timesteps=16,
                 **kwargs):
        super(SequentialModel, self).__init__()
        self.num_pitches = num_pitches
        self.timesteps = timesteps
        self.num_voices = 1
        self.kwargs = kwargs
        self.model_type = model_type

        # load or create dataset
        assert os.path.exists(dataset_name)
        self.dataset_filepath = dataset_name

        # self.dataset = pickle.load(open(self.dataset_filepath, 'rb'))
        # self.metadatas = self.dataset[-1]
        self.generator_unitary = self.generator(phase='all', batch_size=1)

        # unique filepath
        od = collections.OrderedDict(sorted(self.kwargs.items()))
        params_string = '_'.join(
            ['='.join(x) for x in zip(od.keys(), map(str, od.values()))])

        self.filepath = os.path.join(PACKAGE_DIR,
                                     f'models/{self.model_type}_'
                                     f'{params_string}.h5'
                                     )

    def __str__(self):
        return self.filepath

    def hidden_repr(self, *input):
        NotImplementedError

    def forward(self, *input):
        NotImplementedError

    def generator(self, batch_size, phase, percentage_train=0.8, **kwargs):
        raise NotImplementedError

    def hidden_init(self, batch_size, hidden_size, volatile=False) -> Variable:
        hidden = (Variable(torch.rand(self.num_layers, batch_size,
                                      hidden_size).cuda(),
                           volatile=volatile),
                  Variable(torch.rand(self.num_layers, batch_size,
                                      hidden_size).cuda(),
                           volatile=volatile))
        return hidden

    def no_data_init(self, seq_length,
                     batch_size,
                     input_size,
                     volatile=False) -> Variable:
        no_data = Variable(torch.cat(
            (torch.zeros(seq_length - 1, batch_size, input_size),
             torch.ones(seq_length - 1, batch_size, 1)), 2).cuda(),
                           volatile=volatile)
        return no_data

    def save(self):
        torch.save(self.state_dict(), self.filepath)

    def load(self):
        state_dict = torch.load(self.filepath)
        self.load_state_dict(state_dict)

    def find_nearests(self, target_seq, num_elements=1000, show_results=False):
        """
        :param target_seq: 1D torch.LongTensor
        :type target_seq:

        """
        max_dist = 100000

        generator_unitary = self.generator(batch_size=1,
                                           phase='all')

        if target_seq is None:
            target_chorale, _, _, _ = next(generator_unitary)
        else:
            target_chorale = target_seq[None, :]
        target_chorale_cuda = Variable(target_chorale.cuda())

        hidden_repr = self.hidden_repr(target_chorale_cuda)
        intermediate_results = []
        for next_element in tqdm(islice(generator_unitary, num_elements)):

            # to cuda Variable
            next_element_cuda = [
                Variable(tensor.cuda())
                for tensor in next_element
            ]
            input_cuda, _, _, _ = next_element_cuda

            hidden_repr_gen = self.hidden_repr(input_cuda)

            # dist = spearmanr(hidden_repr[0], hidden_repr_gen[0])[0]
            dist = spearman_rho(variable2numpy(hidden_repr[0]),
                                variable2numpy(hidden_repr_gen[0]))

            # if dist > max_dist:
            if dist < max_dist:
                # if dist < 100:
                max_dist = dist
                min_chorale = variable2numpy(input_cuda)
                print(max_dist)

                intermediate_results.append(min_chorale)

        intermediate_results.append(min_chorale)
        print(max_dist)
        if show_results:
            # concat all results
            nearest_chorale = np.concatenate(
                [target_chorale[SOP_INDEX].numpy()] +
                [np.array(
                    nearest_chorales[SOP_INDEX])
                    for nearest_chorales in
                    intermediate_results],
                axis=0)

            _, _, index2notes, note2indexes, _ = pickle.load(open(
                self.dataset_filepath, 'rb'))
            score_nearest = indexed_seq_to_score(nearest_chorale,
                                                 index2notes[SOP_INDEX],
                                                 note2indexes[SOP_INDEX])
            score_nearest.show()
        return min_chorale, intermediate_results

    def show_mean_distance_matrix(self, chorale_index=0, timesteps=32,
                                  show_plotly=False):

        (X,
         voice_ids,
         index2notes,
         note2indexes,
         metadatas) = pickle.load(open(self.dataset_filepath, 'rb'))

        # input:
        chorale_transpositions = X[chorale_index]
        sequence_length = len(chorale_transpositions[0][SEQ][SOP_INDEX])
        num_chunks = sequence_length - timesteps

        mat = np.zeros((num_chunks, len(chorale_transpositions),
                        len(chorale_transpositions)))

        for time_index in tqdm(range(num_chunks)):
            for index_transposition_1, chorale_1 in enumerate(
                    chorale_transpositions):

                hidden_repr_1 = self.hidden_repr(
                    numpy2variable(chorale_1[SEQ][:, time_index:
                    time_index +
                    timesteps])
                )[0]
                for index_transposition_2, chorale_2 in enumerate(
                        chorale_transpositions):

                    hidden_repr_2 = self.hidden_repr(
                        numpy2variable(chorale_2[SEQ][:, time_index:
                        time_index +
                        timesteps])
                    )[0]

                    mat[time_index,
                        index_transposition_1,
                        index_transposition_2] = spearman_rho(hidden_repr_1,
                                                              hidden_repr_2)

        mean = np.mean(mat, axis=0)
        std = np.std(mat, axis=0)
        if show_plotly:
            import seaborn as sns
            from matplotlib import pyplot as plt
            fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
            sns.heatmap(mean, ax=ax1)
            sns.heatmap(std, ax=ax2)
        else:
            print(mean)
            print(std)

class InvariantDistance(SequentialModel):
    def __init__(self,
                 dataset_name,
                 timesteps,
                 num_units_lstm=256,
                 num_pitches=None,
                 dropout_prob=0.2,
                 num_layers=1,
                 embedding_dim=16,
                 non_linearity=None,
                 model_type='invariant_distance',
                 **kwargs):
        super(InvariantDistance, self).__init__(model_type,
                                                dataset_name,
                                                num_pitches,
                                                timesteps,
                                                **kwargs)
        self.non_linearity = non_linearity
        self.embedding_dim = embedding_dim
        self.num_lstm_units = num_units_lstm
        self.num_layers = num_layers

        # Parameters
        self.embedding = nn.Embedding(num_embeddings=num_pitches,
                                      embedding_dim=self.embedding_dim)
        self.lstm_e = nn.LSTM(input_size=self.embedding_dim,
                              hidden_size=self.num_lstm_units,
                              num_layers=self.num_layers,
                              dropout=dropout_prob)

        # TODO add repr_size
        self.lstm_d = nn.LSTM(
            input_size=self.num_lstm_units + self.embedding_dim + 1,
            hidden_size=self.num_lstm_units,
            num_layers=self.num_layers,
            dropout=dropout_prob)

        self.linear_out = nn.Linear(in_features=self.num_lstm_units,
                                    out_features=num_pitches)

    def forward(self, inputs, first_note):
        # todo add hidden to inputs?
        batch_size, seq_length = inputs[0].size()
        assert seq_length == self.timesteps

        hidden_reprs = [self.hidden_repr(input)
                        for input in inputs]

        hidden_repr_1, hidden_repr_2 = hidden_reprs
        diff = hidden_repr_1 - hidden_repr_2
        mean = (hidden_repr_1 + hidden_repr_2) / 2

        weights, softmax = self.decode(hidden_repr=mean,
                                       first_note=first_note,
                                       sequence_length=seq_length)
        return weights, softmax, diff

    def hidden_repr(self, input):
        """

        :param input:(batch_size, seq_length)
        :type input:
        :param hidden:
        :type hidden:
        :return: (batch_size, hidden_repr_dim)
        :rtype:
        """
        # todo check dimensions
        batch_size, _ = input.size()
        embedding = self.embedding(input)
        embedding_time_major = torch.transpose(
            embedding, 0, 1)

        hidden = self.hidden_init(batch_size, self.num_lstm_units)

        outputs_lstm, _ = self.lstm_e(embedding_time_major, hidden)
        last_output = outputs_lstm[-1]

        if self.non_linearity:
            NotImplementedError
        else:
            return last_output

    def decode(self, hidden_repr, first_note, sequence_length):
        """

        :param hidden_repr:(batch_size, hidden_repr_size)
        :type hidden_repr:
        :return:(timestep, batch_size, num_pitches)
        :rtype:
        """
        # transform input as seq
        batch_size, hidden_repr_size = hidden_repr.size()
        embedded_note = self.embedding(first_note)

        input = torch.cat((hidden_repr,
                           embedded_note),
                          1)

        no_data = self.no_data_init(seq_length=sequence_length,
                                    batch_size=batch_size,
                                    input_size=input.size()[1])

        hidden = self.hidden_init(batch_size,
                                  self.num_lstm_units
                                  )

        # input_size = input.size()
        # input_as_seq = torch.cat((
        #     input[None, :, :],
        #     Variable(torch.zeros((self.timesteps - 1,) + input_size).cuda())
        # ), 0)

        input_extended = torch.cat(
            (input[None, :, :], Variable(torch.zeros(1, batch_size, 1).cuda(
            ))), 2)
        input_as_seq = torch.cat(
            [input_extended, no_data], 0)

        output_lstm, _ = self.lstm_d(input_as_seq, hidden)
        weights = [self.linear_out(time_slice) for time_slice in output_lstm]
        softmax = [F.softmax(time_slice) for time_slice in weights]
        softmax = torch.cat(softmax)
        softmax = softmax.view(self.timesteps, batch_size, self.num_pitches)

        weights = torch.cat(weights)
        weights = weights.view(self.timesteps, batch_size, self.num_pitches)
        return weights, softmax

    def numpy_indexed2chunk(self, indexed_chorale,
                            start_symbols,
                            end_symbols,
                            time_index):
        """

        :param indexed_chorale:
        :type indexed_chorale:
        :param start_symbols: used for padding
        :type start_symbols:
        :param end_symbols:
        :type end_symbols:
        :return: random onehot chunk of size (timesteps, num_pitches)
        :rtype:
        """
        # pad with start and end symbols
        padding_dimensions = (self.timesteps,)

        input_seq = np.concatenate(
            (np.full(padding_dimensions, start_symbols),
             indexed_chorale[SOP_INDEX],
             np.full(padding_dimensions, end_symbols)),
            axis=0).astype(np.long)

        return input_seq[time_index: time_index + self.timesteps]

    def generator(self, batch_size, phase, percentage_train=0.8, **kwargs):
        """

        :param batch_size:
        :param phase:
        :param percentage_train:
        :return:
        """
        # TODO effective timesteps?

        X, voice_ids, index2notes, note2indexes, metadatas = pickle.load(
            open(self.dataset_filepath, 'rb'))

        start_symbols = np.array(list(
            map(lambda note2index: note2index[START_SYMBOL],
                note2indexes)))[SOP_INDEX]
        end_symbols = np.array(list(
            map(lambda note2index: note2index[END_SYMBOL], note2indexes)))[
            SOP_INDEX]

        # Set chorale_indices
        if phase == 'train':
            chorale_indices = np.arange(int(len(X) * percentage_train))
        elif phase == 'test':
            chorale_indices = np.arange(int(len(X) * percentage_train), len(X))
        elif phase == 'all':
            chorale_indices = np.arange(int(len(X)))
        else:
            NotImplementedError

        first_notes = []
        sequences = [[] for _ in range(3)]

        batch = 0

        while True:
            chorale_index = np.random.choice(chorale_indices)
            chorales = X[chorale_index]
            if len(chorales) < 3:
                continue
            transposition_indexes = np.random.choice(len(chorales),
                                                     replace=False, size=3)

            # there's padding of size timesteps before and after
            chorale_length = len(
                chorales[0][0][SOP_INDEX]) + 2 * self.timesteps
            time_index = np.random.randint(0,
                                           chorale_length - self.timesteps)
            for seq_index, seq_list in enumerate(sequences):
                transposition_index = transposition_indexes[seq_index]
                indexed_chorale, _, offset = np.array(
                    chorales[transposition_index])
                # padding of size timesteps
                chunk = self.numpy_indexed2chunk(indexed_chorale,
                                                 start_symbols,
                                                 end_symbols,
                                                 time_index)
                seq_list.append(chunk)

            # find first note symbol of output seq (last sequence in sequences)
            first_note_index_output_seq = \
                first_note_index(chunk,
                                 time_index_start=0,
                                 time_index_end=self.timesteps,
                                 note2index=note2indexes[SOP_INDEX])
            first_notes.append(first_note_index_output_seq)

            batch += 1
            # if there is a full batch
            if batch == batch_size:
                torch_sequences = [torch.from_numpy(np.array(chunk))
                                   for chunk in sequences]
                torch_first_notes = torch.from_numpy(np.array(first_notes,
                                                              dtype=np.long))

                # input_1, input_2, first_note, output
                next_element = (torch_sequences[0],
                                torch_sequences[1],
                                torch_first_notes,
                                torch_sequences[2],
                                )
                yield next_element

                batch = 0
                first_notes = []
                sequences = [[] for _ in range(3)]


class InvariantDistanceRelu(InvariantDistance):
    def __init__(self,
                 dataset_name,
                 timesteps,
                 num_units_lstm=256,
                 num_pitches=None,
                 dropout_prob=0.2,
                 num_layers=1,
                 embedding_dim=16,
                 non_linearity=None,
                 mlp_hidden_size=None,
                 model_type='invariant_distance_relu',
                 **kwargs):
        super(InvariantDistanceRelu, self).__init__(model_type=model_type,
                                                    dataset_name=dataset_name,
                                                    timesteps=timesteps,
                                                    num_units_lstm=num_units_lstm,
                                                    num_pitches=num_pitches,
                                                    dropout_prob=dropout_prob,
                                                    num_layers=num_layers,
                                                    embedding_dim=embedding_dim,
                                                    non_linearity=non_linearity,
                                                    **kwargs
                                                    )
        self.linear_1_mlp = nn.Linear(in_features=self.num_lstm_units,
                                      out_features=mlp_hidden_size)
        self.linear_2_mlp = nn.Linear(in_features=mlp_hidden_size,
                                      out_features=self.num_lstm_units)

    def hidden_repr(self, input):
        hidden_no_relu = super(InvariantDistanceRelu, self).hidden_repr(
            input=input)
        # return self.linear_1_mlp(hidden_no_relu)
        return hidden_no_relu
        # return F.elu(self.linear_1_mlp(hidden_no_relu))

    def forward(self, inputs, first_note):
        # todo add hidden to inputs?
        batch_size, seq_length = inputs[0].size()
        assert seq_length == self.timesteps

        hidden_reprs_relu = [self.hidden_repr(input)
                             for input in inputs]

        hidden_repr_relu_1, hidden_repr_relu_2 = hidden_reprs_relu
        diff_relu = hidden_repr_relu_1 - hidden_repr_relu_2

        # hidden_reprs = [self.linear_2_mlp(hidden_repr_relu)
        #                 for hidden_repr_relu in hidden_reprs_relu]
        hidden_reprs = hidden_reprs_relu

        hidden_repr_1, hidden_repr_2 = hidden_reprs
        # diff = hidden_repr_1 - hidden_repr_2
        mean = (hidden_repr_1 + hidden_repr_2) / 2

        weights, softmax = self.decode(hidden_repr=mean,
                                       first_note=first_note,
                                       sequence_length=seq_length)
        return weights, softmax, diff_relu

# TODO specific hidden_repr size
