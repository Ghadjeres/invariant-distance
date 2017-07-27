import collections
import heapq
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
from deepPermutations.losses import crossentropy_loss, accuracy
from deepPermutations.permutation_distance import spearman_rho, \
    distance_from_name
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm


class SequentialModel(nn.Module):
    """
    Base class for distance models
    """

    def __init__(self,
                 model_type: str,
                 dataset_name: str,
                 num_pitches=None,
                 timesteps=16,
                 **kwargs
                 ):
        """

        :param model_type:
        :type model_type:
        :param dataset_name:
        :type dataset_name:
        :param num_pitches:
        :type num_pitches:
        :param timesteps:
        :type timesteps:
        :param non_linearity:
        :type non_linearity:
        :param kwargs: must only include parameters used in filepath string
        :type kwargs:
        """
        super(SequentialModel, self).__init__()
        self.num_pitches = num_pitches
        self.timesteps = timesteps
        self.num_voices = 1

        self.model_type = model_type

        # load or create dataset
        assert os.path.exists(dataset_name)
        self.dataset_filepath = dataset_name

        # load generator
        self.generator_unitary = self.generator(phase='all', batch_size=1)

        od = collections.OrderedDict(sorted(kwargs.items()))
        params_string = '_'.join(
            ['='.join(x) for x in zip(od.keys(), map(str, od.values()))])

        model_dir = os.path.join(PACKAGE_DIR,
                                 f'models',
                                 f'{self.model_type}'
                                 )
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        self.filepath = os.path.join(model_dir,
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

    def target_seq(self):
        X, voice_ids, index2notes, note2indexes, metadatas = pickle.load(
            open(self.dataset_filepath, 'rb'))
        return torch.from_numpy(
            np.array(X[10][2][0][SOP_INDEX][96:96 + self.timesteps])).long()

    def generator_dataset(self, phase, percentage_train=0.8,
                          **kwargs):
        """

        :param batch_size:
        :param phase:
        :param percentage_train:
        :return:
        """
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

        for chorale_index in chorale_indices:
            chorales = X[chorale_index]
            for chorale in chorales:
                indexed_chorale, _, offset = np.array(
                    chorale)

                # there's padding of size timesteps before and after
                chorale_length = len(
                    indexed_chorale[SOP_INDEX]) + 2 * self.timesteps

                padding_dimensions = (self.timesteps,)

                indexed_chorale = np.concatenate(
                    (np.full(padding_dimensions, start_symbols),
                     indexed_chorale[SOP_INDEX],
                     np.full(padding_dimensions, end_symbols)),
                    axis=0).astype(np.long)

                for time_index in range(chorale_length - self.timesteps):
                    chunk = indexed_chorale[
                            time_index: time_index + self.timesteps]

                    torch_chunk = torch.from_numpy(np.array(chunk))
                    yield torch_chunk

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
        self.eval()
        generator_unitary = self.generator(batch_size=1,
                                           phase='all')

        if target_seq is None:
            target_chorale, _, _, _ = next(generator_unitary)
        else:
            target_chorale = target_seq[None, :]
        target_chorale_cuda = Variable(target_chorale.cuda())

        hidden_repr = self.hidden_repr(target_chorale_cuda)
        intermediate_results = []
        for id, next_element in tqdm(enumerate(islice(generator_unitary,
                                                      num_elements))):
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

            # # if dist > max_dist:
            # if dist < max_dist:
            #     # if dist < 100:
            #     max_dist = dist
            #     min_chorale = variable2numpy(input_cuda)
            #     print(max_dist)
            #     intermediate_results.append(min_chorale)

            chorale = variable2numpy(input_cuda)

            heapq.heappush(intermediate_results,
                           (dist, id, chorale)
                           )

        if show_results:
            nearest_chorales = [
                chorale
                for dist, id, chorale in heapq.nsmallest(20,
                                                         intermediate_results,
                                                         key=lambda e: e[0])]
            #
            # for dist, chorale in heapq.nsmallest(20,
            #                                      intermediate_results,
            #                                      key=lambda e: e[0]):
            #     print(dist)

            # concat all results
            nearest_chorale = np.concatenate(
                [target_chorale[SOP_INDEX].numpy()] +
                [np.array(
                    nearest_chorale[SOP_INDEX])
                    for nearest_chorale in
                    nearest_chorales],
                axis=0)

            _, _, index2notes, note2indexes, _ = pickle.load(open(
                self.dataset_filepath, 'rb'))
            score_nearest = indexed_seq_to_score(nearest_chorale,
                                                 index2notes[SOP_INDEX],
                                                 note2indexes[SOP_INDEX])
            score_nearest.show()
        return nearest_chorales

    def find_nearests_all(self,
                          target_seq,
                          num_nearests=20,
                          show_results=False,
                          distance='spearman'):
        """
        :param distance:
        :type distance:
        :param show_results:
        :type show_results:
        :param num_nearests:
        :type num_nearests:
        :param target_seq: 1D torch.LongTensor
        :type target_seq:

        """
        self.eval()
        generator_dataset = self.generator_dataset(
            phase='all')

        generator_unitary = self.generator(
            batch_size=1,
            phase='all')

        if target_seq is None:
            next_element = next(generator_unitary)
            target_chorale = next_element[0]
        else:
            target_chorale = target_seq[None, :]
        target_chorale_cuda = Variable(target_chorale.cuda())

        hidden_repr = self.hidden_repr(target_chorale_cuda)
        intermediate_results = []
        for id, chunk in tqdm(enumerate(generator_dataset)):

            if distance == 'edit':
                dist = distance_from_name(distance)(
                    target_chorale.numpy()[SOP_INDEX],
                    variable2numpy(chunk)[SOP_INDEX]
                )
            else:
                # to cuda Variable
                input_cuda = Variable(chunk.cuda())[None, :]

                hidden_repr_gen = self.hidden_repr(input_cuda)

                dist = distance_from_name(distance)(
                    variable2numpy(hidden_repr[0]),
                    variable2numpy(hidden_repr_gen[0])
                )

                chorale = variable2numpy(input_cuda)

                heapq.heappush(intermediate_results,
                               (dist, id, chorale)
                               )

            if len(intermediate_results) > 512:
                intermediate_results = intermediate_results[:512]
                heapq.heapify(intermediate_results)

            if id > 200000:
                break

        if show_results:
            nearest_chorales = [
                chorale
                for dist, id, chorale in heapq.nsmallest(num_nearests,
                                                         intermediate_results,
                                                         key=lambda e: e[0])
            ]

            for dist, id, chorale in heapq.nsmallest(num_nearests,
                                                     intermediate_results,
                                                     key=lambda e: e[0]):
                print(dist)

            # concat all results
            nearest_chorale = np.concatenate(
                [target_chorale[SOP_INDEX].numpy()] +
                [np.array(
                    nearest_chorale[SOP_INDEX])
                    for nearest_chorale in
                    nearest_chorales],
                axis=0)

            _, _, index2notes, note2indexes, _ = pickle.load(open(
                self.dataset_filepath, 'rb'))
            score_nearest = indexed_seq_to_score(nearest_chorale,
                                                 index2notes[SOP_INDEX],
                                                 note2indexes[SOP_INDEX])
            score_nearest.show()
        return nearest_chorales

    def show_mean_distance_matrix(self, chorale_index=0,
                                  show_plot=False):
        self.eval()
        (X,
         voice_ids,
         index2notes,
         note2indexes,
         metadatas) = pickle.load(open(self.dataset_filepath, 'rb'))

        # input:
        chorale_transpositions = X[chorale_index]
        sequence_length = len(chorale_transpositions[0][SEQ][SOP_INDEX])
        num_chunks = sequence_length - self.timesteps

        mat = np.zeros((num_chunks, len(chorale_transpositions),
                        len(chorale_transpositions)))

        # todo remove
        for time_index in tqdm(range(num_chunks)):
            for index_transposition_1, chorale_1 in enumerate(
                    chorale_transpositions):

                hidden_repr_1 = self.hidden_repr(
                    numpy2variable(chorale_1[SEQ][:, time_index:
                    time_index +
                    self.timesteps],
                                   dtype=np.long,
                                   volatile=True)
                )[0]

                hidden_repr_1 = variable2numpy(hidden_repr_1)

                for index_transposition_2, chorale_2 in enumerate(
                        chorale_transpositions):
                    hidden_repr_2 = self.hidden_repr(
                        numpy2variable(chorale_2[SEQ][:, time_index:
                        time_index +
                        self.timesteps],
                                       dtype=np.long,
                                       volatile=True
                                       )
                    )[0]
                    hidden_repr_2 = variable2numpy(hidden_repr_2)

                    mat[time_index,
                        index_transposition_1,
                        index_transposition_2] = spearman_rho(hidden_repr_1,
                                                              hidden_repr_2)

        mean = np.mean(mat, axis=0)
        std = np.std(mat, axis=0)
        if show_plot:
            import seaborn as sns
            from matplotlib import pyplot as plt
            # plt.ion()
            fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
            sns.heatmap(mean, ax=ax1)
            sns.heatmap(std, ax=ax2)
            sns.plt.show()
        else:
            print(mean)
            print(std)

    def compute_stats(self,
                      chorale_index=0,
                      num_elements=1000,
                      timesteps=32,
                      export_filename='results/stats.csv'):
        """

        :param num_elements:
        :return: features produced by generator that achieve minimum distance
        """
        self.eval()
        (X,
         voice_ids,
         index2notes,
         note2indexes,
         metadatas) = pickle.load(open(self.dataset_filepath, 'rb'))

        # input:
        chorale_transpositions = X[chorale_index]
        # sequence_length = len(chorale_transpositions[0][SEQ][SOP_INDEX])
        generator_unitary = self.generator_unitary

        distance_diff = []
        distance_same = []

        for _ in tqdm(range(num_elements)):
            # different sequences:
            hidden_repr_1 = self.hidden_repr(
                Variable(next(generator_unitary)[0].cuda(),
                         volatile=True)
            )[0]
            hidden_repr_2 = self.hidden_repr(
                Variable(next(generator_unitary)[0].cuda(),
                         volatile=True)
            )[0]

            hidden_repr_1 = variable2numpy(hidden_repr_1)
            hidden_repr_2 = variable2numpy(hidden_repr_2)

            distance_diff.append(spearman_rho(hidden_repr_1, hidden_repr_2))

            # same sequence up to transposition
            chorale_transpositions = np.random.choice(X)

            chorale_length = len(chorale_transpositions[0][SEQ][SOP_INDEX])
            time_index = np.random.choice(chorale_length - timesteps)
            transposition_index_1, transposition_index_2 = np.random.choice(
                len(chorale_transpositions), size=2)
            transposition_1 = chorale_transpositions[transposition_index_1]
            transposition_2 = chorale_transpositions[transposition_index_2]

            hidden_repr_1 = self.hidden_repr(
                numpy2variable(transposition_1[SEQ][:, time_index: time_index +
                                                                   self.timesteps],
                               dtype=np.long,
                               volatile=True)
            )[0]
            hidden_repr_2 = self.hidden_repr(
                numpy2variable(transposition_2[SEQ][:, time_index: time_index +
                                                                   self.timesteps],
                               dtype=np.long,
                               volatile=True)
            )[0]

            hidden_repr_1 = variable2numpy(hidden_repr_1)
            hidden_repr_2 = variable2numpy(hidden_repr_2)

            distance_same.append(spearman_rho(hidden_repr_1, hidden_repr_2))

        hist_data = [distance_diff, distance_same]

        # save into file
        with open(os.path.join(PACKAGE_DIR,
                               export_filename), 'w') as f:
            f.write(f'distance, label\n')
            for i, label in enumerate(['random', 'transposition']):
                for d in hist_data[i]:
                    f.write(f'{d}, {label}\n')

        import seaborn as sns
        from matplotlib import pyplot as plt
        sns.set()
        fig, ax = plt.subplots()
        for a in hist_data:
            sns.distplot(a, ax=ax, kde=True)
        sns.plt.show()  # TODO  clean refactor


def non_linearity_from_name(non_linearity):
    if not non_linearity:
        return lambda x: x
    else:
        if non_linearity == 'ReLU':
            return nn.ReLU()
        else:
            raise NotImplementedError


class Distance(SequentialModel):
    def __init__(self,
                 dataset_name,
                 timesteps,
                 num_pitches=None,
                 num_lstm_units=256,
                 dropout_prob=0.2,
                 input_dropout=0.1,
                 num_layers=1,
                 embedding_dim=16,
                 non_linearity=None,
                 ):
        model_type = 'distance'
        super(Distance, self).__init__(model_type,
                                       dataset_name,
                                       num_pitches,
                                       timesteps,
                                       non_linearity=non_linearity,
                                       dropout_prob=dropout_prob,
                                       input_dropout=input_dropout,
                                       num_lstm_units=num_lstm_units,
                                       num_layers=num_layers,
                                       embedding_dim=embedding_dim
                                       )
        self.embedding_dim = embedding_dim
        self.num_lstm_units = num_lstm_units
        self.num_layers = num_layers

        # Parameters
        self.embedding = nn.Embedding(num_embeddings=num_pitches,
                                      embedding_dim=self.embedding_dim)
        self.lstm_e = nn.LSTM(input_size=self.embedding_dim,
                              hidden_size=self.num_lstm_units,
                              num_layers=self.num_layers,
                              dropout=dropout_prob)

        self.lstm_d = nn.LSTM(
            input_size=self.num_lstm_units + 1,
            hidden_size=self.num_lstm_units,
            num_layers=self.num_layers,
            dropout=dropout_prob)

        self.linear_out = nn.Linear(in_features=self.num_lstm_units,
                                    out_features=num_pitches)

        # common layers
        self.non_linearity = non_linearity_from_name(non_linearity)

        self.input_dropout_layer = nn.Dropout2d(input_dropout)

    def forward(self, input):
        batch_size, seq_length = input.size()
        assert seq_length == self.timesteps

        hidden_repr = self.hidden_repr(input)
        weights, softmax = self.decode(hidden_repr=hidden_repr,
                                       sequence_length=seq_length)
        return weights, softmax

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

        # input_dropout
        embedding_time_major = self.input_dropout_layer(
            embedding_time_major[:, :, :, None]
        )[:, :, :, 0]

        hidden = self.hidden_init(batch_size, self.num_lstm_units)

        outputs_lstm, _ = self.lstm_e(embedding_time_major, hidden)
        last_output = outputs_lstm[-1]

        # ReLU or nothing
        last_output = self.non_linearity(last_output)

        return last_output

    def decode(self, hidden_repr, sequence_length):
        """

        :param hidden_repr:(batch_size, hidden_repr_size)
        :type hidden_repr:
        :return:(timestep, batch_size, num_pitches)
        :rtype:
        """
        # transform input as seq
        batch_size, hidden_repr_size = hidden_repr.size()

        input = hidden_repr

        no_data = self.no_data_init(seq_length=sequence_length,
                                    batch_size=batch_size,
                                    input_size=input.size()[1])

        hidden = self.hidden_init(batch_size,
                                  self.num_lstm_units
                                  )

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

    def loss_functions(self, next_element):
        # to cuda
        next_element_cuda = [
            Variable(tensor.cuda())
            for tensor in next_element
        ]
        input, output = next_element_cuda
        weights, softmax = self.forward(input)

        # mce_loss
        mce_loss = crossentropy_loss(weights, output)
        reg = Variable(torch.zeros(1).cuda())
        acc = accuracy(weights, output)
        return mce_loss, reg, acc

    def generator(self, batch_size, phase, percentage_train=0.8, **kwargs):
        """

        :param batch_size:
        :param phase:
        :param percentage_train:
        :return:
        """
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

        batch = 0
        chunks = []

        while True:
            chorale_index = np.random.choice(chorale_indices)
            chorales = X[chorale_index]

            transposition_index = np.random.randint(len(chorales))

            # there's padding of size timesteps before and after
            chorale_length = len(
                chorales[0][0][SOP_INDEX]) + 2 * self.timesteps
            time_index = np.random.randint(0,
                                           chorale_length - self.timesteps)

            indexed_chorale, _, offset = np.array(
                chorales[transposition_index])
            # padding of size timesteps
            chunk = self.numpy_indexed2chunk(indexed_chorale,
                                             start_symbols,
                                             end_symbols,
                                             time_index)
            chunks.append(chunk)

            batch += 1
            # if there is a full batch
            if batch == batch_size:
                torch_sequences = torch.from_numpy(np.array(chunks))

                next_element = (torch_sequences,
                                torch_sequences
                                )
                yield next_element
                batch = 0
                chunks = []


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
                 input_dropout=0,
                 **kwargs):
        super(InvariantDistance, self).__init__(model_type,
                                                dataset_name,
                                                num_pitches,
                                                timesteps,
                                                input_dropout=input_dropout,
                                                **kwargs)
        self.non_linearity = non_linearity
        self.embedding_dim = embedding_dim
        self.num_lstm_units = num_units_lstm
        self.num_layers = num_layers
        self.input_dropout = input_dropout

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

        self.input_dropout = nn.Dropout(input_dropout)

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
        # todo input_dropout?
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

        self.relu = nn.ReLU()

    def hidden_repr(self, input):
        hidden_no_relu = super(InvariantDistanceRelu, self).hidden_repr(
            input=input)
        return self.relu(hidden_no_relu)

    def forward(self, inputs, first_note):
        # todo add hidden to inputs?
        batch_size, seq_length = inputs[0].size()
        assert seq_length == self.timesteps

        hidden_reprs_relu = [self.hidden_repr(input)
                             for input in inputs]

        hidden_repr_relu_1, hidden_repr_relu_2 = hidden_reprs_relu
        diff_relu = hidden_repr_relu_1 - hidden_repr_relu_2

        hidden_reprs = hidden_reprs_relu

        hidden_repr_1, hidden_repr_2 = hidden_reprs

        mean = (hidden_repr_1 + hidden_repr_2) / 2

        weights, softmax = self.decode(hidden_repr=mean,
                                       first_note=first_note,
                                       sequence_length=seq_length)
        return weights, softmax, diff_relu

# TODO specific hidden_repr size
