import argparse
import collections
import os
import pickle
from itertools import islice
from typing import Optional

import plotly.figure_factory as ff
import plotly.graph_objs as go
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.models import load_model

from DeepBach.metadata import *
from tqdm import tqdm

from .data_preprocessing import to_pitch_class, SOP_INDEX, to_onehot, \
    indexed_seq_to_score
from .data_utils import initialization, all_features, all_metadatas, \
    START_SYMBOL, \
    END_SYMBOL, \
    indexed_chorale_to_score, chorale_to_onehot, SLUR_SYMBOL
from .distance_models import seq2seq, invariant_seq2seq, \
    invariant_seq2seq_NN, \
    invariant_absolute_seq2seq, \
    invariant_absolute_seq2seq_reg, l2_norm, l1_norm, \
    invariant_absolute_seq2seq_reg_mean, \
    invariant_absolute_seq2seq_reg_mean_relu
from .generation_models import deepbach_skip_sop
from .permutation import spearman_rho, chorale_onehot_to_indexed_chorale


class SequentialModel:
    def __init__(self, name: str, model_type: str, dataset_name: str,
                 timesteps=16, **kwargs):
        self.timesteps = timesteps
        self.name = name
        self.num_voices = 1

        # load or create dataset
        self.dataset_filepath = 'datasets/' + dataset_name + '.pickle'

        # datasets are in 'transpose' format
        if not os.path.exists(self.dataset_filepath):
            self.create_dataset()
        self.dataset = pickle.load(open(self.dataset_filepath, 'rb'))
        self.metadatas = self.dataset[-1]

        self.generator_unitary = self.generator(phase='all', batch_size=1)

        od = collections.OrderedDict(sorted(kwargs.items()))
        params_string = '_'.join(
            ['='.join(x) for x in zip(od.keys(), map(str, od.values()))])
        # load or create model
        self.filepath = 'models/' + model_type + '/' + self.name + '_' + params_string + '.h5'
        if os.path.exists(self.filepath):
            self.model = load_model(self.filepath,
                                    custom_objects={'l2_norm': l2_norm,
                                                    'l1_norm': l1_norm})  # type: Model
            print('Model ' + self.filepath + ' loaded')
        else:
            self.model = self.create_model(**kwargs)  # type: Model
            print('Model ' + self.filepath + ' created')
        self.model.summary()
        self.hidden_repr_model = self._hidden_repr_model()

    def generator(self, batch_size, phase, percentage_train=0.8, **kwargs):
        raise NotImplementedError

    def create_dataset(self):
        raise NotImplementedError

    def create_model(self, **kwargs):
        raise NotImplementedError

    def _hidden_repr_model(self):
        return None

    def train(self, batch_size, nb_epochs, steps_per_epoch, validation_steps,
              overwrite=True, **kwargs):
        # generators
        generator_train = self.generator(phase='train', batch_size=batch_size,
                                         **kwargs)
        generator_test = self.generator(phase='test', batch_size=batch_size,
                                        **kwargs)
        # train
        self.model.fit_generator(generator=generator_train,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=nb_epochs,
                                 validation_data=generator_test,
                                 validation_steps=validation_steps,
                                 callbacks=[
                                     # EarlyStopping(min_delta=0.001, patience=5),
                                     TensorBoard(log_dir='logs')]
                                 )
        self.model.save(self.filepath, overwrite=overwrite)
        print('Model saved')
        # update hidden_repr_model
        self.hidden_repr_model = self._hidden_repr_model()


class InvariantDistanceModel(SequentialModel):
    def __init__(self, name: str, dataset_name: str, timesteps=16,
                 dropout: Optional[float] = 0.2,
                 **kwargs):
        # self.transpose_dataset_filepath = 'datasets/transpose/bach_sop.pickle'
        # if not os.path.exists(self.transpose_dataset_filepath):
        #     raise NotImplementedError
        # self.transpose_dataset = pickle.load(open(self.transpose_dataset_filepath, 'rb'))
        self.max_offset = 15
        self.kwargs = kwargs
        super(InvariantDistanceModel, self).__init__(name=name,
                                                     model_type='invariant_distance',
                                                     dataset_name=dataset_name,
                                                     timesteps=timesteps,
                                                     **kwargs)
        # index2notes is a list!
        (_, voice_ids, self.index2notes,
         self.note2indexes, self.metadatas) = pickle.load(
            open(self.dataset_filepath, 'rb'))

    def create_dataset(self):
        raise NotImplementedError

    def find_nearests(self, features, num_elements=1000, show_results=False,
                      effective_timestep=None):
        """

        :param num_elements:
        :param features:
        :return: features produced by generator that achieve minimum distance
        """
        # max_dist = -10
        max_dist = 100000
        num_pitches = list(map(len, self.dataset[3]))

        if effective_timestep:
            generator_unitary = self.generator(batch_size=1, phase='all',
                                               effective_timestep=effective_timestep)
        else:
            generator_unitary = self.generator_unitary

        min_chorale = features
        hidden_repr = self.hidden_repr_model.predict(features, batch_size=1)
        intermediate_results = []
        for features_gen in tqdm(islice(generator_unitary, num_elements)):
            hidden_repr_gen = self.hidden_repr_model.predict(features_gen[0],
                                                             batch_size=1)

            # dist = spearmanr(hidden_repr[0], hidden_repr_gen[0])[0]
            dist = spearman_rho(hidden_repr[0], hidden_repr_gen[0])

            # if dist > max_dist:
            if dist < max_dist:
                # if dist < 100:
                max_dist = dist
                min_chorale = features_gen
                print(max_dist)

                intermediate_results.append(min_chorale)

        intermediate_results.append(min_chorale)
        print(max_dist)
        if show_results:
            # concat all results
            nearest_chorale = np.concatenate([features['input_seq'][0]] +
                                             [np.array(
                                                 nearest_chorale_inputs[0][
                                                     'input_seq'][0])
                                                 for nearest_chorale_inputs in
                                                 intermediate_results],
                                             axis=0)

            # create score nearest
            nearest_chorale_seq = chorale_onehot_to_indexed_chorale(
                onehot_chorale=nearest_chorale,
                num_pitches=num_pitches,
                time_major=False)

            score_nearest = indexed_seq_to_score(nearest_chorale_seq[0],
                                                 self.index2notes[SOP_INDEX],
                                                 self.note2indexes[SOP_INDEX])
            score_nearest.show()
        return min_chorale, intermediate_results

    def generator(self, batch_size, phase, percentage_train=0.8,
                  effective_timestep=None):
        """

        :param batch_size:
        :param phase:
        :param percentage_train:
        :param effective_timestep:
        :return:
        """
        X, voice_ids, index2notes, note2indexes, metadatas = pickle.load(
            open(self.dataset_filepath, 'rb'))
        num_pitches = list(map(lambda x: len(x), index2notes))[SOP_INDEX]
        num_voices = len(voice_ids)
        # Set chorale_indices
        if phase == 'train':
            chorale_indices = np.arange(int(len(X) * percentage_train))
        elif phase == 'test':
            chorale_indices = np.arange(int(len(X) * percentage_train), len(X))
        elif phase == 'all':
            chorale_indices = np.arange(int(len(X)))
        else:
            NotImplementedError

        input_seqs = []
        transposed_input_seqs = []
        output_seqs = []
        input_offsets = []
        first_notes = []

        batch = 0

        while True:
            chorale_index = np.random.choice(chorale_indices)
            chorales = X[chorale_index]
            if len(chorales) == 1:
                continue
            if not effective_timestep:
                effective_timestep = np.random.randint(min(8, self.timesteps),
                                                       self.timesteps + 1)

            (transposition_index_1,
             transposition_index_2,
             transposition_index_3) = np.random.choice(len(chorales),
                                                       replace=True, size=3)

            input_seq, _, offset_1 = np.array(chorales[transposition_index_1])
            transposed_input_seq, _, _ = np.array(chorales[
                                                      transposition_index_2])
            output_seq, _, offset_3 = np.array(chorales[transposition_index_3])

            # pad with start and end symbols
            padding_dimensions = (self.timesteps,)
            start_symbols = np.array(list(
                map(lambda note2index: note2index[START_SYMBOL],
                    note2indexes)))[SOP_INDEX]
            end_symbols = np.array(list(
                map(lambda note2index: note2index[END_SYMBOL], note2indexes)))[
                SOP_INDEX]

            input_seq = np.concatenate(
                (np.full(padding_dimensions, start_symbols),
                 input_seq[SOP_INDEX],
                 np.full(padding_dimensions, end_symbols)),
                axis=0)
            transposed_input_seq = np.concatenate(
                (np.full(padding_dimensions, start_symbols),
                 transposed_input_seq[SOP_INDEX],
                 np.full(padding_dimensions, end_symbols)),
                axis=0)
            output_seq = np.concatenate(
                (np.full(padding_dimensions, start_symbols),
                 output_seq[SOP_INDEX],
                 np.full(padding_dimensions, end_symbols)),
                axis=0)

            chorale_length = len(input_seq)
            time_index = np.random.randint(0, chorale_length - self.timesteps)

            # find first note symbol
            first_note_index_output_seq = first_note_index(output_seq,
                                                           time_index_start=time_index + self.timesteps - effective_timestep,
                                                           time_index_end=time_index + self.timesteps,
                                                           note2index=
                                                           note2indexes[
                                                               SOP_INDEX])

            # to onehot
            input_seq = np.array(
                list(map(lambda x: to_onehot(x, num_pitches), input_seq)))
            transposed_input_seq = np.array(
                list(map(lambda x: to_onehot(x, num_pitches),
                         transposed_input_seq)))
            output_seq = np.array(
                list(map(lambda x: to_onehot(x, num_pitches), output_seq)))

            first_note_output_seq = to_onehot(first_note_index_output_seq,
                                              num_indexes=num_pitches)

            input_seqs.append(
                input_seq[time_index: time_index + self.timesteps, :])
            transposed_input_seqs.append(
                transposed_input_seq[time_index: time_index + self.timesteps,
                :])
            output_seqs.append(
                output_seq[time_index: time_index + self.timesteps, :])
            input_offsets.append(
                to_onehot(offset_3 - offset_1 + self.max_offset,
                          self.max_offset * 2))
            first_notes.append(first_note_output_seq)

            batch += 1

            # if there is a full batch
            if batch == batch_size:
                input_seqs = np.array(input_seqs)
                transposed_input_seqs = np.array(transposed_input_seqs)
                output_seqs = np.array(output_seqs)
                input_offsets = np.array(input_offsets)
                first_notes = np.array(first_notes)

                # pad with -1 for elements outside effective timestep
                input_seqs = np.concatenate(
                    [np.full((batch_size, self.timesteps - effective_timestep,
                              input_seqs.shape[-1]),
                             -1),
                     input_seqs[:, -effective_timestep:, :]
                     ],
                    axis=1)
                transposed_input_seqs = np.concatenate(
                    [np.full((batch_size, self.timesteps - effective_timestep,
                              transposed_input_seqs.shape[-1]),
                             -1),
                     transposed_input_seqs[:, -effective_timestep:, :]
                     ],
                    axis=1)
                output_seqs = np.concatenate([
                    output_seqs[:, -effective_timestep:, :],
                    np.zeros((batch_size, self.timesteps - effective_timestep,
                              output_seqs.shape[-1]))
                ],
                    axis=1)

                # make output = input for debug and input_offsets = 0
                # output_seqs = np.concatenate([
                #     input_seqs[:, -effective_timestep:, :],
                #     np.zeros((batch_size, self.timesteps - effective_timestep, output_seqs.shape[-1]))
                # ],
                #     axis=1)
                # input_offsets = np.zeros((batch_size, self.max_offset * 2))

                next_element = ({'input_seq': input_seqs,
                                 'input_offset': input_offsets,
                                 'first_note': first_notes,
                                 'transposed_input': transposed_input_seqs,
                                 },
                                {
                                    'output_seq': output_seqs,
                                    'diff_hidden_repr': np.zeros((batch_size,
                                                                  1)),
                                }
                                )

                yield next_element

                batch = 0

                input_seqs = []
                transposed_input_seqs = []
                output_seqs = []
                input_offsets = []
                first_notes = []

    def show_preds(self, num_elements=10, effective_timestep=None,
                   absolute_transposition=False):
        """

        :param num_elements:
        :param features:
        :return: features produced by generator that achieve minimum distance
        """
        # max_dist = -10
        num_pitches = list(map(len, self.dataset[3]))

        if effective_timestep:
            generator_unitary = self.generator(batch_size=1, phase='test',
                                               effective_timestep=effective_timestep)
        else:
            generator_unitary = self.generator_unitary

        res = []

        for features_gen in tqdm(islice(generator_unitary, num_elements)):
            onehot_in = features_gen[0]['input_seq'][0]
            indexed_input_seq = chorale_onehot_to_indexed_chorale(
                onehot_chorale=onehot_in,
                num_pitches=num_pitches,
                time_major=False)
            if not absolute_transposition:
                print(np.argmax(features_gen[0]['input_offset']) - 15)
            res.extend(indexed_input_seq[0])

            seq_pred = self.model.predict(features_gen[0], batch_size=1)[0]
            res.extend(list(map(np.argmax, seq_pred[SOP_INDEX])))

        # seq = np.concatenate([features['input_seq'][0]] +
        #                      [np.array(nearest_chorale_inputs[0]['input_seq'][0])
        #                       for nearest_chorale_inputs in intermediate_results],
        #                      axis=0)
        # create score nearest

        score_nearest = indexed_seq_to_score(res, self.index2notes[SOP_INDEX],
                                             self.note2indexes[SOP_INDEX])
        score_nearest.show()

    def show_all_absolute_preds(self, effective_timestep=None):
        """
        :return: features produced by generator that achieve minimum distance
        """
        num_pitches = list(map(len, self.dataset[3]))

        if effective_timestep:
            generator_unitary = self.generator(batch_size=1, phase='test',
                                               effective_timestep=effective_timestep)
        else:
            generator_unitary = self.generator_unitary

        res = []
        # target seq
        features_gen = next(generator_unitary)
        onehot_in = features_gen[0]['input_seq'][0]
        indexed_input_seq = chorale_onehot_to_indexed_chorale(
            onehot_chorale=onehot_in,
            num_pitches=num_pitches,
            time_major=False)

        res.extend(indexed_input_seq[0])

        # predict all transpositions
        for note_index in self.index2notes[SOP_INDEX]:
            features_in = features_gen[0]

            # set first note
            features_in['first_note'] = to_onehot(note_index, len(
                self.index2notes[SOP_INDEX]))[None, :]
            seq_pred = self.model.predict(features_in, batch_size=1)[0]
            res.extend(list(map(np.argmax, seq_pred[SOP_INDEX])))

        # seq = np.concatenate([features['input_seq'][0]] +
        #                      [np.array(nearest_chorale_inputs[0]['input_seq'][0])
        #                       for nearest_chorale_inputs in intermediate_results],
        #                      axis=0)
        # create score nearest

        score_nearest = indexed_seq_to_score(res, self.index2notes[SOP_INDEX],
                                             self.note2indexes[SOP_INDEX])
        score_nearest.show()

    def test_transpose_out_of_bounds(self, effective_timestep=None):
        """

        :param num_elements:
        :param features:
        :return: features produced by generator that achieve minimum distance
        """
        # max_dist = -10
        num_pitches = list(map(len, self.dataset[3]))

        if effective_timestep:
            generator_unitary = self.generator(batch_size=1, phase='test',
                                               effective_timestep=effective_timestep)
        else:
            generator_unitary = self.generator_unitary

        # input:
        features_gen = next(generator_unitary)
        onehot_in = features_gen[0]['input_seq'][0]

        # add input for printing
        res = []
        # indexed_input_seq = chorale_onehot_to_indexed_chorale(
        #     onehot_chorale=onehot_in,
        #     num_pitches=num_pitches,
        #     time_major=False)
        res.extend(list(map(np.argmax, onehot_in)))
        for offset in range(0, 2 * self.max_offset):
            features_gen[0]['input_offset'] = to_onehot(offset,
                                                        2 * self.max_offset)[
                                              None, :]
            seq_pred = self.model.predict(features_gen[0], batch_size=1)[0]
            res.extend(list(map(np.argmax, seq_pred)))

        # seq = np.concatenate([features['input_seq'][0]] +
        #                      [np.array(nearest_chorale_inputs[0]['input_seq'][0])
        #                       for nearest_chorale_inputs in intermediate_results],
        #                      axis=0)
        # create score nearest

        score_nearest = indexed_seq_to_score(res, self.index2notes[SOP_INDEX],
                                             self.note2indexes[SOP_INDEX])
        score_nearest.show()

    def show_distance_matrix(self, chorale_index=0, time_index=0, timesteps=32,
                             show_plotly=False):
        """

        :param num_elements:
        :param features:
        :return: features produced by generator that achieve minimum distance
        """
        X, voice_ids, self.index2notes, self.note2indexes, self.metadatas = pickle.load(
            open(self.dataset_filepath, 'rb'))
        num_pitches = list(map(len, self.dataset[3]))

        # input:
        mat = np.zeros((len(X[chorale_index]), len(X[chorale_index])))

        for index_transposition_1, chorale_1 in enumerate(X[chorale_index]):

            hidden_repr_1 = self.hidden_repr_model.predict({
                'input_seq': chorale_to_onehot(
                    chorale_1[SEQ][:, time_index: time_index + timesteps],
                    num_pitches=num_pitches,
                    time_major=False)[None, :, :]
            })[0]
            for index_transposition_2, chorale_2 in enumerate(
                    X[chorale_index]):
                hidden_repr_2 = self.hidden_repr_model.predict({
                    'input_seq': chorale_to_onehot(
                        chorale_2[SEQ][:, time_index: time_index + timesteps],
                        num_pitches=num_pitches,
                        time_major=False)[None, :, :]
                })[0]
                mat[
                    index_transposition_1, index_transposition_2] = spearman_rho(
                    hidden_repr_1, hidden_repr_2)

        if show_plotly:
            from plotly.offline import plot

            data = [
                go.Heatmap(
                    z=mat
                )
            ]
            plot(data, filename='distance_matrix')

        else:
            print(mat)

    def show_mean_distance_matrix(self, chorale_index=0, timesteps=32,
                                  show_plotly=False):
        """

        :param num_elements:
        :param features:
        :return: features produced by generator that achieve minimum distance
        """
        X, voice_ids, self.index2notes, self.note2indexes, self.metadatas = pickle.load(
            open(self.dataset_filepath, 'rb'))
        num_pitches = list(map(len, self.dataset[3]))

        # input:
        chorale_transpositions = X[chorale_index]
        sequence_length = len(chorale_transpositions[0][SEQ][SOP_INDEX])
        num_chunks = sequence_length - timesteps

        mat = np.zeros((num_chunks, len(chorale_transpositions),
                        len(chorale_transpositions)))

        for time_index in tqdm(range(num_chunks)):
            for index_transposition_1, chorale_1 in enumerate(
                    chorale_transpositions):

                hidden_repr_1 = self.hidden_repr_model.predict({
                    'input_seq': chorale_to_onehot(
                        chorale_1[SEQ][:, time_index: time_index + timesteps],
                        num_pitches=num_pitches,
                        time_major=False)[None, :, :]
                })[0]
                for index_transposition_2, chorale_2 in enumerate(
                        chorale_transpositions):
                    hidden_repr_2 = self.hidden_repr_model.predict({
                        'input_seq': chorale_to_onehot(chorale_2[SEQ][:,
                                                       time_index: time_index + timesteps],
                                                       num_pitches=num_pitches,
                                                       time_major=False)[None,
                                     :, :]
                    })[0]
                    mat[
                        time_index, index_transposition_1, index_transposition_2] = spearman_rho(
                        hidden_repr_1,
                        hidden_repr_2)

        mean = np.mean(mat, axis=0)
        std = np.std(mat, axis=0)
        if show_plotly:
            from plotly.offline import plot
            from plotly import tools
            fig = tools.make_subplots(rows=1, cols=2,
                                      subplot_titles=('Mean', 'Std'))
            fig.append_trace(go.Heatmap(
                z=mean
            ), 1, 1)
            fig.append_trace(
                go.Heatmap(
                    z=std
                ), 1, 2)
            plot(fig, filename='mean_distance_matrix.html')

        else:
            print(mean)
            print(std)

    def create_model(self, **kwargs):
        if self.name == 'seq2seq':
            inputs, labels = next(self.generator_unitary)
            num_features = inputs['input_seq'].shape[-1]
            return invariant_seq2seq(self.timesteps, num_features=num_features,
                                     num_units_lstm=512,
                                     dropout_prob=self.dropout, masking=False,
                                     num_offsets=self.max_offset * 2)
        if self.name == 'seq2seq_masking':
            inputs, labels = next(self.generator_unitary)
            num_features = inputs['input_seq'].shape[-1]
            return invariant_seq2seq(self.timesteps, num_features=num_features,
                                     num_units_lstm=512,
                                     dropout_prob=self.dropout,
                                     masking=True,
                                     num_offsets=self.max_offset * 2)
        if self.name == 'seq2seq_NN':
            inputs, labels = next(self.generator_unitary)
            num_features = inputs['input_seq'].shape[-1]
            return invariant_seq2seq_NN(self.timesteps,
                                        num_features=num_features,
                                        num_units_lstm=512,
                                        dropout_prob=self.dropout,
                                        masking=False,
                                        num_offsets=self.max_offset * 2)
        if self.name == 'seq2seq_large':
            inputs, labels = next(self.generator_unitary)
            num_features = inputs['input_seq'].shape[-1]
            return invariant_seq2seq(self.timesteps, num_features=num_features,
                                     num_units_lstm=128,
                                     dropout_prob=self.dropout,
                                     num_layers=2,
                                     masking=True,
                                     num_offsets=self.max_offset * 2)
        if self.name == 'seq2seq_absolute_invariant':
            num_units_lstm = kwargs['num_units_lstm']
            num_layers = kwargs['num_layers']
            dropout_prob = kwargs['dropout_prob']
            inputs, labels = next(self.generator_unitary)
            num_features = inputs['input_seq'].shape[-1]
            num_pitches = inputs['first_note'].shape[-1]
            return invariant_absolute_seq2seq(self.timesteps,
                                              num_features=num_features,
                                              num_units_lstm=num_units_lstm,
                                              dropout_prob=dropout_prob,
                                              num_layers=num_layers,
                                              masking=False,
                                              num_pitches=num_pitches)

        if self.name == 'seq2seq_absolute_invariant_reg':
            if kwargs['reg'] == 'l1':
                reg = l1_norm
            elif kwargs['reg'] == 'l2':
                reg = l2_norm
            else:
                reg = None

            num_units_lstm = kwargs['num_units_lstm']
            num_layers = kwargs['num_layers']
            dropout_prob = kwargs['dropout_prob']
            inputs, labels = next(self.generator_unitary)
            num_features = inputs['input_seq'].shape[-1]
            num_pitches = inputs['first_note'].shape[-1]
            return invariant_absolute_seq2seq_reg(self.timesteps,
                                                  num_features=num_features,
                                                  num_units_lstm=num_units_lstm,
                                                  dropout_prob=dropout_prob,
                                                  num_layers=num_layers,
                                                  masking=False,
                                                  num_pitches=num_pitches,
                                                  reg=reg)

        if self.name == 'seq2seq_absolute_invariant_reg_mean':
            if kwargs['reg'] == 'l1':
                reg = l1_norm
            elif kwargs['reg'] == 'l2':
                reg = l2_norm
            else:
                reg = None

            num_units_lstm = kwargs['num_units_lstm']
            num_layers = kwargs['num_layers']
            dropout_prob = kwargs['dropout_prob']
            inputs, labels = next(self.generator_unitary)
            num_features = inputs['input_seq'].shape[-1]
            num_pitches = inputs['first_note'].shape[-1]
            return invariant_absolute_seq2seq_reg_mean(self.timesteps,
                                                       num_features=num_features,
                                                       num_units_lstm=num_units_lstm,
                                                       dropout_prob=dropout_prob,
                                                       num_layers=num_layers,
                                                       masking=False,
                                                       num_pitches=num_pitches,
                                                       reg=reg)

        if self.name == 'seq2seq_absolute_invariant_reg_mean_relu':
            if kwargs['reg'] == 'l1':
                reg = l1_norm
            elif kwargs['reg'] == 'l2':
                reg = l2_norm
            else:
                reg = None

            num_units_lstm = kwargs['num_units_lstm']
            num_layers = kwargs['num_layers']
            dropout_prob = kwargs['dropout_prob']
            inputs, labels = next(self.generator_unitary)
            num_features = inputs['input_seq'].shape[-1]
            num_pitches = inputs['first_note'].shape[-1]
            return invariant_absolute_seq2seq_reg_mean_relu(self.timesteps,
                                                            num_features=num_features,
                                                            num_units_lstm=num_units_lstm,
                                                            dropout_prob=dropout_prob,
                                                            num_layers=num_layers,
                                                            masking=False,
                                                            num_pitches=num_pitches,
                                                            reg=reg)

    def _hidden_repr_model(self):
        return Model(self.model.get_layer('input_seq').input,
                     self.model.get_layer('hidden_repr').output)

    def batched_distances(self, seq, list_of_constraints, num_pitches,
                          time_indexes, voice_index):
        num_voices = seq.shape[-1]
        batched_dists = []
        is_dist = []
        no_dist = np.full((num_pitches[voice_index],), -1)

        for batch_index, time_index in enumerate(time_indexes):
            in_constraint = False
            for constraint in list_of_constraints:
                # tweak probas with distance model
                a, b = constraint['first_interval']
                if a <= time_index < b:
                    # assert constraint['first_interval'][1] - constraint['first_interval'][0] <= self.timesteps
                    # assert constraint['second_interval'][1] - constraint['second_interval'][0] <= self.timesteps
                    in_constraint = True

                    # test all indexes
                    # create batch for parallel updates
                    batch_current = []
                    for pitch_index in range(num_pitches[voice_index]):
                        # compute current

                        offset = b - self.timesteps
                        # todo change all_features to return only one segment
                        seq_current = seq[
                                      offset: offset + self.timesteps + self.timesteps,
                                      :].copy()
                        seq_current[
                            time_index - offset, voice_index] = pitch_index

                        left_features_current, _, _, _ = all_features(
                            seq_current,
                            voice_index=voice_index,
                            time_index=self.timesteps,
                            timesteps=self.timesteps,
                            num_pitches=num_pitches,
                            num_voices=num_voices)

                        # pad beginning of sequence with -1
                        left_features_current[:a - offset, :] = -1
                        input_features_current = {
                            'input_seq': left_features_current,
                        }

                        batch_current.append(input_features_current)

                    # convert input_features
                    batch_current = {
                        key: np.array(
                            [input_features[key] for input_features in
                             batch_current])
                        for key in batch_current[0].keys()
                    }

                    # create target
                    # todo use argument constraint['transpose']
                    if 'second_interval' in constraint:
                        raise NotImplementedError
                    elif 'target' in constraint:
                        batch_target = constraint['target']
                    else:
                        raise NotImplementedError

                    # predict all hidden_repr in parallel
                    hidden_reprs_current = self.hidden_repr_model.predict(
                        batch_current, batch_size=len(batch_current))
                    hidden_repr_target = \
                        self.hidden_repr_model.predict(batch_target,
                                                       batch_size=len(
                                                           batch_target))[0]
                    # predict all distances
                    dists = np.array(list(map(
                        lambda hidden_repr_current: spearman_rho(
                            hidden_repr_current, hidden_repr_target),
                        hidden_reprs_current)))

                    batched_dists.append(dists)
                    is_dist.append(True)

                    # print(np.min(dists), np.max(dists))
                    break
            if not in_constraint:
                batched_dists.append(no_dist)
                is_dist.append(False)
        return np.array(batched_dists), np.array(is_dist)

    def compute_stats(self, chorale_index=0, num_elements=1000, timesteps=32,
                      effective_timesteps=32):
        """

        :param num_elements:
        :return: features produced by generator that achieve minimum distance
        """
        X, voice_ids, self.index2notes, self.note2indexes, self.metadatas = pickle.load(
            open(self.dataset_filepath, 'rb'))
        num_pitches = list(map(len, self.dataset[3]))

        # input:
        chorale_transpositions = X[chorale_index]
        sequence_length = len(chorale_transpositions[0][SEQ][SOP_INDEX])
        if effective_timesteps:
            generator_unitary = self.generator(batch_size=1, phase='all',
                                               effective_timestep=effective_timesteps)
        else:
            generator_unitary = self.generator_unitary
        distance_diff = []
        distance_same = []

        for _ in tqdm(range(num_elements)):
            hidden_repr_1 = self.hidden_repr_model.predict(
                next(generator_unitary)[0]['input_seq'])[0]
            hidden_repr_2 = self.hidden_repr_model.predict(
                next(generator_unitary)[0]['input_seq'])[0]

            distance_diff.append(spearman_rho(hidden_repr_1, hidden_repr_2))

            chorale_transpositions = np.random.choice(X)

            chorale_length = len(chorale_transpositions[0][SEQ][SOP_INDEX])
            time_index = np.random.choice(chorale_length - timesteps)
            transposition_index_1, transposition_index_2 = np.random.choice(
                len(chorale_transpositions), size=2)
            transposition_1 = chorale_transpositions[transposition_index_1]
            transposition_2 = chorale_transpositions[transposition_index_2]
            hidden_repr_1 = self.hidden_repr_model.predict({
                'input_seq': chorale_to_onehot(transposition_1[SEQ][:,
                                               time_index: time_index + timesteps],
                                               num_pitches=num_pitches,
                                               time_major=False)[None, :, :]
            })[0]
            hidden_repr_2 = self.hidden_repr_model.predict({
                'input_seq': chorale_to_onehot(transposition_2[SEQ][:,
                                               time_index: time_index + timesteps],
                                               num_pitches=num_pitches,
                                               time_major=False)[None, :, :]
            })[0]

            distance_same.append(spearman_rho(hidden_repr_1, hidden_repr_2))

        hist_data = [distance_diff, distance_same]

        from plotly.offline import plot
        fig = ff.create_distplot(hist_data=hist_data,
                                 group_labels=['Random', 'Transposition'])

        plot(fig, filename='stats.html')


class DistanceModel(SequentialModel):
    def __init__(self, name: str, dataset_name: str, timesteps=16,
                 dropout=0.2):
        self.dropout = dropout
        self.max_offset = 15
        super(DistanceModel, self).__init__(name=name, model_type='distance',
                                            dataset_name=dataset_name,
                                            timesteps=timesteps)

    def create_dataset(self):
        raise NotImplementedError

    # def generator(self, batch_size, phase, percentage_train=0.8):
    #     return (
    #         ({'input_seq': left_features,
    #           },
    #          {'output_seq': left_features})
    #         for (
    #         (left_features, central_features, right_features),
    #         (left_metas, central_metas, right_metas),
    #         labels
    #     )
    #         in generator_from_raw_dataset(batch_size=batch_size,
    #                                       timesteps=self.timesteps,
    #                                       voice_index=SOP_INDEX,
    #                                       phase=phase,
    #                                       percentage_train=percentage_train,
    #                                       pickled_dataset=self.dataset_filepath)
    #     )

    def find_nearests(self, features, num_elements=1000, show_results=False,
                      effective_timestep=None):
        """

        :param num_elements:
        :param features:
        :return: features produced by generator that achieve minimum distance
        """
        # max_dist = -10
        max_dist = 1000
        num_pitches = list(map(len, self.dataset[3]))

        if effective_timestep:
            generator_unitary = self.generator(batch_size=1, phase='all',
                                               effective_timestep=effective_timestep)
        else:
            generator_unitary = self.generator_unitary

        min_chorale = features
        hidden_repr = self.hidden_repr_model.predict(features, batch_size=1)
        intermediate_results = []
        for features_gen in tqdm(islice(generator_unitary, num_elements)):
            hidden_repr_gen = self.hidden_repr_model.predict(features_gen[0],
                                                             batch_size=1)

            # dist = spearmanr(hidden_repr[0], hidden_repr_gen[0])[0]
            dist = spearman_rho(hidden_repr[0], hidden_repr_gen[0])

            # if dist > max_dist:
            if dist < max_dist:
                max_dist = dist
                min_chorale = features_gen
                print(max_dist)

                if dist < 200:
                    intermediate_results.append(min_chorale)

        intermediate_results.append(min_chorale)
        print(max_dist)
        if show_results:
            # concat all results
            nearest_chorale = np.concatenate([features['input_seq'][0]] +
                                             [np.array(
                                                 nearest_chorale_inputs[0][
                                                     'input_seq'][0])
                                                 for nearest_chorale_inputs in
                                                 intermediate_results],
                                             axis=0)

            # create score nearest
            nearest_chorale_seq = chorale_onehot_to_indexed_chorale(
                onehot_chorale=nearest_chorale,
                num_pitches=num_pitches,
                time_major=False)

            score_nearest = indexed_chorale_to_score(nearest_chorale_seq,
                                                     pickled_dataset=self.dataset_filepath)
            score_nearest.show()
        return min_chorale, intermediate_results

    def generator(self, batch_size, phase, percentage_train=0.8,
                  effective_timestep=None):
        raise NotImplementedError

    def create_model(self, **kwargs):
        if self.name == 'seq2seq':
            inputs, labels = next(self.generator_unitary)
            num_features = inputs['input_seq'].shape[-1]
            return seq2seq(self.timesteps, num_features=num_features,
                           num_units_lstm=128, dropout_prob=self.dropout,
                           masking=False)
        if self.name == 'seq2seq_masking':
            inputs, labels = next(self.generator_unitary)
            num_features = inputs['input_seq'].shape[-1]
            return seq2seq(self.timesteps, num_features=num_features,
                           num_units_lstm=128, dropout_prob=self.dropout,
                           masking=True)

    def batched_distances(self, seq, list_of_constraints, num_pitches,
                          time_indexes, voice_index):
        num_voices = seq.shape[-1]
        batched_dists = []
        is_dist = []
        no_dist = np.full((num_pitches[voice_index],), -1)

        for batch_index, time_index in enumerate(time_indexes):
            in_constraint = False
            for constraint in list_of_constraints:
                # tweak probas with distance model
                a, b = constraint['first_interval']
                if a <= time_index < b:
                    # assert constraint['first_interval'][1] - constraint['first_interval'][0] <= self.timesteps
                    # assert constraint['second_interval'][1] - constraint['second_interval'][0] <= self.timesteps
                    in_constraint = True

                    # test all indexes
                    # create batch for parallel updates
                    batch_current = []
                    for pitch_index in range(num_pitches[voice_index]):
                        # compute current

                        offset = b - self.timesteps
                        # todo change all_features to return only one segment
                        seq_current = seq[
                                      offset: offset + self.timesteps + self.timesteps,
                                      :].copy()
                        seq_current[
                            time_index - offset, voice_index] = pitch_index

                        left_features_current, _, _, _ = all_features(
                            seq_current,
                            voice_index=voice_index,
                            time_index=self.timesteps,
                            timesteps=self.timesteps,
                            num_pitches=num_pitches,
                            num_voices=num_voices)

                        # pad beginning of sequence with -1
                        left_features_current[:a - offset, :] = -1
                        input_features_current = {
                            'input_seq': left_features_current,
                        }

                        batch_current.append(input_features_current)

                    # convert input_features
                    batch_current = {
                        key: np.array(
                            [input_features[key] for input_features in
                             batch_current])
                        for key in batch_current[0].keys()
                    }

                    # create target
                    # todo use argument constraint['transpose']
                    a, b = constraint['second_interval']
                    offset = b - self.timesteps
                    # todo change all_features to return only one segment
                    seq_target = seq[
                                 offset: offset + self.timesteps + self.timesteps,
                                 :]  # .copy()

                    left_features_target, _, _, _ = all_features(seq_target,
                                                                 voice_index=voice_index,
                                                                 time_index=self.timesteps,
                                                                 timesteps=self.timesteps,
                                                                 num_pitches=num_pitches,
                                                                 num_voices=num_voices)
                    # pad beginning of sequence with -1
                    left_features_target[: a - offset, :] = -1
                    batch_target = np.array([left_features_target])

                    # predict all hidden_repr in parallel
                    hidden_reprs_current = self.model.predict(batch_current,
                                                              batch_size=len(
                                                                  batch_current))
                    hidden_repr_target = self.model.predict(batch_target,
                                                            batch_size=len(
                                                                batch_target))[
                        0]
                    # predict all distances
                    dists = np.array(list(map(
                        lambda hidden_repr_current: spearman_rho(
                            hidden_repr_current, hidden_repr_target),
                        hidden_reprs_current)))

                    batched_dists.append(dists)
                    is_dist.append(True)

                    # print(np.min(dists), np.max(dists))
                    break
            if not in_constraint:
                batched_dists.append(no_dist)
                is_dist.append(False)
        return np.array(batched_dists), np.array(is_dist)


class ShapeModel(DistanceModel):
    def __init__(self, name: str, dataset_name: str, timesteps=16,
                 dropout=0.2):
        self.max_offset = 15

        # first load or create dataset
        self.dataset_filepath = 'datasets/' + dataset_name + '.pickle'
        self.dataset = pickle.load(open(self.dataset_filepath, 'rb'))
        self.shape_dataset_filepath = 'datasets/shape/' + dataset_name + '.pickle'
        if not os.path.exists(self.shape_dataset_filepath):
            self.create_dataset()
            # self.shape_dataset = pickle.load(open(self.shape_dataset_filepath, 'rb'))

        self.dropout = dropout
        super(DistanceModel, self).__init__(name=name, model_type='shape',
                                            dataset_name=dataset_name,
                                            timesteps=timesteps)

        _, self.pc_index2pc, self.pc2pc_index, self.note_index2pc_index = pickle.load(
            open(self.shape_dataset_filepath, 'rb'))
        # todo refactor: compute inverse mapping when creating dataset
        # compute inverse mapping
        self.pc_index2note_indexes = {}
        for note_index, pc_index in self.note_index2pc_index.items():
            if pc_index not in self.pc_index2note_indexes:
                self.pc_index2note_indexes[pc_index] = [note_index]
            else:
                list_of_pcs = self.pc_index2note_indexes[pc_index]
                list_of_pcs.append(note_index)

    def _hidden_repr_model(self):
        return Model(self.model.get_layer('input_seq').input,
                     self.model.get_layer('hidden_repr').output)

    def create_dataset(self):
        X, voice_ids, index2notes, note2indexes, metadatas = self.dataset

        # compute new dictionnaries
        # Todo ONLY ONE VOICE
        pc2pc_index = {}
        pc_index2pc = {}
        note_index2pc_index = {}
        all_pcs = list(
            set(map(to_pitch_class, note2indexes[SOP_INDEX].keys())))

        for pc_index, pc in enumerate(all_pcs):
            pc2pc_index.update({pc: pc_index})
            pc_index2pc.update({pc_index: pc})

        for note_index, note in index2notes[SOP_INDEX].items():
            note_index2pc_index.update({note_index:
                                            pc2pc_index[to_pitch_class(note)]})

        X_pc = []
        for chorale_transpositions in X:
            c_pc = []
            offset = -1
            first_note = None
            # c is a (chorale, metas, offset) tuple
            for c in chorale_transpositions:
                pcs = np.array([list(
                    map(lambda note_index: note_index2pc_index[note_index],
                        c[0][SOP_INDEX]))])
                # Recopy metas and recompute good offset
                # todo problem if starts with start/end/slur symbol
                if first_note != pcs[0][SOP_INDEX]:
                    first_note = pcs[0][SOP_INDEX]
                    offset += 1
                    c_pc.append(
                        (pcs, c[1], offset)
                    )
            X_pc.append(c_pc)

        shape_dataset = X_pc, pc_index2pc, pc2pc_index, note_index2pc_index
        pickle.dump(shape_dataset, open(self.shape_dataset_filepath, 'wb'))

    def show_preds(self, num_elements=10, effective_timestep=None):
        """

        :param num_elements:
        :param features:
        :return: features produced by generator that achieve minimum distance
        """
        # max_dist = -10
        num_pitches = [len(self.pc2pc_index)]

        if effective_timestep:
            generator_unitary = self.generator(batch_size=1, phase='test',
                                               effective_timestep=effective_timestep)
        else:
            generator_unitary = self.generator_unitary

        res = []

        for features_gen in tqdm(islice(generator_unitary, num_elements)):
            seq_pred = self.model.predict(features_gen[0], batch_size=1)
            onehot_in = features_gen[0]['input_seq'][0]
            indexed_input_seq = chorale_onehot_to_indexed_chorale(
                onehot_chorale=onehot_in,
                num_pitches=num_pitches,
                time_major=False)
            print(np.argmax(features_gen[0]['input_offset']) - 15)
            res.extend(indexed_input_seq[0])
            res.extend(list(map(np.argmax, seq_pred[0])))

        # seq = np.concatenate([features['input_seq'][0]] +
        #                      [np.array(nearest_chorale_inputs[0]['input_seq'][0])
        #                       for nearest_chorale_inputs in intermediate_results],
        #                      axis=0)
        # create score nearest

        score_nearest = indexed_seq_to_score(res, self.pc_index2pc,
                                             self.pc2pc_index)
        score_nearest.show()

    def generator(self, batch_size, phase, percentage_train=0.8,
                  effective_timestep=None):
        """

        :param batch_size:
        :param phase:
        :param percentage_train:
        :param effective_timestep:
        :return:
        """
        # load shape dataset
        X_pc, pc_index2pc, pc2pc_index, note_index2pc_index = pickle.load(
            open(self.shape_dataset_filepath, 'rb'))

        num_pitches = len(pc_index2pc)
        # Todo only one voice!
        # num_voices = len(voice_ids)
        num_voices = 1
        # Set chorale_indices
        if phase == 'train':
            chorale_indices = np.arange(int(len(X_pc) * percentage_train))
        elif phase == 'test':
            chorale_indices = np.arange(int(len(X_pc) * percentage_train),
                                        len(X_pc))
        elif phase == 'all':
            chorale_indices = np.arange(int(len(X_pc)))
        else:
            NotImplementedError

        input_seqs = []
        output_seqs = []
        input_offsets = []
        first_notes = []

        batch = 0

        while True:
            chorale_index = np.random.choice(chorale_indices)
            chorales = X_pc[chorale_index]
            if len(chorales) == 1:
                continue
            if not effective_timestep:
                effective_timestep = np.random.randint(min(8, self.timesteps),
                                                       self.timesteps + 1)

            transposition_index_1, transposition_index_2 = np.random.choice(
                len(chorales), replace=True, size=2)
            input_seq, _, offset_1 = np.array(chorales[transposition_index_1])
            output_seq, _, offset_2 = np.array(chorales[transposition_index_2])

            # pad with start and end symbols
            padding_dimensions = (self.timesteps,)
            start_symbols = np.array(list(
                map(lambda note2index: note2index[START_SYMBOL],
                    [pc2pc_index])))[SOP_INDEX]
            end_symbols = np.array(list(
                map(lambda note2index: note2index[END_SYMBOL],
                    [pc2pc_index])))[SOP_INDEX]

            input_seq = np.concatenate(
                (np.full(padding_dimensions, start_symbols),
                 input_seq[SOP_INDEX],
                 np.full(padding_dimensions, end_symbols)),
                axis=0)
            output_seq = np.concatenate(
                (np.full(padding_dimensions, start_symbols),
                 output_seq[SOP_INDEX],
                 np.full(padding_dimensions, end_symbols)),
                axis=0)

            chorale_length = len(input_seq)
            time_index = np.random.randint(0, chorale_length - self.timesteps)

            # find first note symbol
            first_note_index_output_seq = first_note_index(output_seq,
                                                           time_index_start=time_index + self.timesteps - effective_timestep,
                                                           time_index_end=time_index + self.timesteps,
                                                           note2index=pc2pc_index)

            # to onehot
            input_seq = np.array(
                list(map(lambda x: to_onehot(x, num_pitches), input_seq)))
            output_seq = np.array(
                list(map(lambda x: to_onehot(x, num_pitches), output_seq)))
            first_note_output_seq = to_onehot(first_note_index_output_seq,
                                              num_indexes=num_pitches)

            input_seqs.append(
                input_seq[time_index: time_index + self.timesteps, :])
            output_seqs.append(
                output_seq[time_index: time_index + self.timesteps, :])
            input_offsets.append(
                to_onehot(offset_2 - offset_1 + self.max_offset,
                          self.max_offset * 2))
            first_notes.append(first_note_output_seq)

            batch += 1

            # if there is a full batch
            if batch == batch_size:
                input_seqs = np.array(input_seqs)
                output_seqs = np.array(output_seqs)
                input_offsets = np.array(input_offsets)
                first_notes = np.array(first_notes)

                # pad with -1 for elements outside effective timestep
                input_seqs = np.concatenate(
                    [np.full((batch_size, self.timesteps - effective_timestep,
                              input_seqs.shape[-1]),
                             -1),
                     input_seqs[:, -effective_timestep:, :]
                     ],
                    axis=1)
                output_seqs = np.concatenate([
                    output_seqs[:, -effective_timestep:, :],
                    np.zeros((batch_size, self.timesteps - effective_timestep,
                              output_seqs.shape[-1]))
                ],
                    axis=1)

                # make output = input for debug and input_offsets = 0
                # output_seqs = np.concatenate([
                #     input_seqs[:, -effective_timestep:, :],
                #     np.zeros((batch_size, self.timesteps - effective_timestep, output_seqs.shape[-1]))
                # ],
                #     axis=1)
                # input_offsets = np.zeros((batch_size, self.max_offset * 2))

                next_element = ({'input_seq': input_seqs,
                                 'input_offset': input_offsets,
                                 'first_note': first_notes,
                                 'transposed_input': output_seqs,
                                 },
                                {
                                    'output_seq': output_seqs,
                                    'diff_repr': np.zeros((batch_size, 1)),
                                }
                                )

                yield next_element

                batch = 0

                input_seqs = []
                output_seqs = []
                input_offsets = []
                first_notes = []

    def generator_full_names(self, batch_size, phase, percentage_train=0.8,
                             effective_timestep=None):
        """

        :param batch_size:
        :param phase:
        :param percentage_train:
        :param effective_timestep:
        :return:
        """
        X, voice_ids, index2notes, note2indexes, metadatas = pickle.load(
            open(self.dataset_filepath, 'rb'))
        num_pitches = list(map(lambda x: len(x), index2notes))[SOP_INDEX]
        num_voices = len(voice_ids)
        # Set chorale_indices
        if phase == 'train':
            chorale_indices = np.arange(int(len(X) * percentage_train))
        elif phase == 'test':
            chorale_indices = np.arange(int(len(X) * percentage_train), len(X))
        elif phase == 'all':
            chorale_indices = np.arange(int(len(X)))
        else:
            NotImplementedError

        input_seqs = []
        output_seqs = []
        input_offsets = []

        batch = 0

        while True:
            chorale_index = np.random.choice(chorale_indices)
            chorales = X[chorale_index]
            if len(chorales) == 1:
                continue
            if not effective_timestep:
                effective_timestep = np.random.randint(min(8, self.timesteps),
                                                       self.timesteps + 1)

            transposition_index_1, transposition_index_2 = np.random.choice(
                len(chorales), replace=True, size=2)
            input_seq, _, offset_1 = np.array(chorales[transposition_index_1])
            output_seq, _, offset_2 = np.array(chorales[transposition_index_2])

            # pad with start and end symbols
            padding_dimensions = (self.timesteps,)
            start_symbols = np.array(list(
                map(lambda note2index: note2index[START_SYMBOL],
                    note2indexes)))[SOP_INDEX]
            end_symbols = np.array(list(
                map(lambda note2index: note2index[END_SYMBOL], note2indexes)))[
                SOP_INDEX]

            input_seq = np.concatenate(
                (np.full(padding_dimensions, start_symbols),
                 input_seq[SOP_INDEX],
                 np.full(padding_dimensions, end_symbols)),
                axis=0)
            output_seq = np.concatenate(
                (np.full(padding_dimensions, start_symbols),
                 output_seq[SOP_INDEX],
                 np.full(padding_dimensions, end_symbols)),
                axis=0)

            # to onehot
            input_seq = np.array(
                list(map(lambda x: to_onehot(x, num_pitches), input_seq)))
            output_seq = np.array(
                list(map(lambda x: to_onehot(x, num_pitches), output_seq)))

            chorale_length = len(input_seq)
            time_index = np.random.randint(0, chorale_length - self.timesteps)

            input_seqs.append(
                input_seq[time_index: time_index + self.timesteps, :])
            output_seqs.append(
                output_seq[time_index: time_index + self.timesteps, :])
            input_offsets.append(
                to_onehot(offset_2 - offset_1 + self.max_offset,
                          self.max_offset * 2))

            batch += 1

            # if there is a full batch
            if batch == batch_size:
                input_seqs = np.array(input_seqs)
                output_seqs = np.array(output_seqs)
                input_offsets = np.array(input_offsets)

                # pad with -1 for elements outside effective timestep
                input_seqs = np.concatenate(
                    [np.full((batch_size, self.timesteps - effective_timestep,
                              input_seqs.shape[-1]),
                             -1),
                     input_seqs[:, -effective_timestep:, :]
                     ],
                    axis=1)
                output_seqs = np.concatenate([
                    output_seqs[:, -effective_timestep:, :],
                    np.zeros((batch_size, self.timesteps - effective_timestep,
                              output_seqs.shape[-1]))
                ],
                    axis=1)

                # make output = input for debug and input_offsets = 0
                # output_seqs = np.concatenate([
                #     input_seqs[:, -effective_timestep:, :],
                #     np.zeros((batch_size, self.timesteps - effective_timestep, output_seqs.shape[-1]))
                # ],
                #     axis=1)
                # input_offsets = np.zeros((batch_size, self.max_offset * 2))

                next_element = ({'input_seq': input_seqs,
                                 'input_offset': input_offsets
                                 },
                                {
                                    'output_seq': output_seqs
                                }
                                )

                yield next_element

                batch = 0

                input_seqs = []
                output_seqs = []
                input_offsets = []

    def create_model(self):
        if self.name == 'seq2seq':
            inputs, labels = next(self.generator_unitary)
            num_features = inputs['input_seq'].shape[-1]
            return invariant_seq2seq(self.timesteps, num_features=num_features,
                                     num_units_lstm=512,
                                     dropout_prob=self.dropout, masking=False,
                                     num_offsets=self.max_offset * 2)
        if self.name == 'seq2seq_masking':
            inputs, labels = next(self.generator_unitary)
            num_features = inputs['input_seq'].shape[-1]
            return invariant_seq2seq(self.timesteps, num_features=num_features,
                                     num_units_lstm=512,
                                     dropout_prob=self.dropout,
                                     masking=True,
                                     num_offsets=self.max_offset * 2)
        if self.name == 'seq2seq_NN':
            inputs, labels = next(self.generator_unitary)
            num_features = inputs['input_seq'].shape[-1]
            return invariant_seq2seq_NN(self.timesteps,
                                        num_features=num_features,
                                        num_units_lstm=512,
                                        dropout_prob=self.dropout,
                                        masking=False,
                                        num_offsets=self.max_offset * 2)
        if self.name == 'seq2seq_large':
            inputs, labels = next(self.generator_unitary)
            num_features = inputs['input_seq'].shape[-1]
            return invariant_seq2seq(self.timesteps, num_features=num_features,
                                     num_units_lstm=128,
                                     dropout_prob=self.dropout,
                                     num_layers=2,
                                     masking=True,
                                     num_offsets=self.max_offset * 2)
        if self.name == 'seq2seq_absolute_invariant_reg':
            inputs, labels = next(self.generator_unitary)
            num_features = inputs['input_seq'].shape[-1]
            num_pitches = inputs['first_note'].shape[-1]
            return invariant_absolute_seq2seq_reg(self.timesteps,
                                                  num_features=num_features,
                                                  num_units_lstm=128,
                                                  dropout_prob=self.dropout,
                                                  num_layers=2,
                                                  masking=False,
                                                  num_pitches=num_pitches)
        if self.name == 'seq2seq_absolute_invariant_reg_l1':
            inputs, labels = next(self.generator_unitary)
            num_features = inputs['input_seq'].shape[-1]
            num_pitches = inputs['first_note'].shape[-1]
            return invariant_absolute_seq2seq_reg(self.timesteps,
                                                  num_features=num_features,
                                                  num_units_lstm=128,
                                                  dropout_prob=self.dropout,
                                                  num_layers=2,
                                                  masking=False,
                                                  num_pitches=num_pitches,
                                                  reg=l1_norm)

    def find_nearests(self, features, num_elements=1000, show_results=False,
                      effective_timestep=None):
        """

        :param num_elements:
        :param features:
        :return: features produced by generator that achieve minimum distance
        """
        # max_dist = -10
        max_dist = 1000
        num_pitches = list(map(len, self.dataset[3]))

        if effective_timestep:
            generator_unitary = self.generator_full_names(batch_size=1,
                                                          phase='all',
                                                          effective_timestep=effective_timestep)
        else:
            generator_unitary = self.generator_full_names(phase='all',
                                                          batch_size=1)

        min_chorale = features
        hidden_repr = self.hidden_repr_model.predict(features, batch_size=1)
        intermediate_results = []
        for features_gen in tqdm(islice(generator_unitary, num_elements)):
            # change features_gen (full_notes into simple names)
            features_gen_full_notes = features_gen
            features_gen = ({'input_seq': onehot_fullname2onehot_pc(
                features_gen_full_notes[0]['input_seq'],
                note_index2pc_index=self.note_index2pc_index,
                num_pc=len(self.pc2pc_index)),
                                'input_offset': features_gen_full_notes[0][
                                    'input_offset']
                            },
                            {
                                'output_seq': onehot_fullname2onehot_pc(
                                    features_gen_full_notes[1]['output_seq'],
                                    note_index2pc_index=self.note_index2pc_index,
                                    num_pc=len(self.pc2pc_index))
                            }
            )

            hidden_repr_gen = self.hidden_repr_model.predict(features_gen[0],
                                                             batch_size=1)

            # dist = spearmanr(hidden_repr[0], hidden_repr_gen[0])[0]
            dist = spearman_rho(hidden_repr[0], hidden_repr_gen[0])

            # if dist < 80:
            #     intermediate_results.append(min_chorale)

            # if dist > max_dist:
            if dist < max_dist:
                max_dist = dist
                min_chorale = features_gen_full_notes
                print(max_dist)

                if dist < 200:
                    intermediate_results.append(min_chorale)

        intermediate_results.append(min_chorale)
        print(max_dist)
        if show_results:
            # print input_shape
            input_shape = indexed_seq_to_score(
                chorale_onehot_to_indexed_chorale(features['input_seq'][0],
                                                  num_pitches=[
                                                      len(self.pc2pc_index)],
                                                  time_major=False
                                                  )[0]
                , self.pc_index2pc,
                self.pc2pc_index)
            input_shape.show()

            # concat all results
            nearest_chorale = np.concatenate(
                [np.array(nearest_chorale_inputs[0]['input_seq'][0])
                 for nearest_chorale_inputs in intermediate_results],
                axis=0)

            # create score nearest
            nearest_chorale_seq = chorale_onehot_to_indexed_chorale(
                onehot_chorale=nearest_chorale,
                num_pitches=num_pitches,
                time_major=False)

            score_nearest = indexed_chorale_to_score(nearest_chorale_seq,
                                                     pickled_dataset=self.dataset_filepath)
            score_nearest.show()
        return min_chorale, intermediate_results

    # TODO refactor put batched_distances into parent class
    def batched_distances(self, seq, list_of_constraints, num_pitches,
                          time_indexes, voice_index):
        num_voices = seq.shape[-1]
        batched_dists = []
        is_dist = []
        no_dist = np.full((num_pitches[voice_index],), -1)

        for batch_index, time_index in enumerate(time_indexes):
            in_constraint = False
            for constraint in list_of_constraints:
                # tweak probas with distance model
                a, b = constraint['first_interval']
                if a <= time_index < b:
                    # assert constraint['first_interval'][1] - constraint['first_interval'][0] <= self.timesteps
                    # assert constraint['second_interval'][1] - constraint['second_interval'][0] <= self.timesteps
                    in_constraint = True

                    # test all indexes
                    # create batch for parallel updates
                    batch_current = []
                    for pitch_index in range(num_pitches[voice_index]):
                        # compute current

                        offset = b - self.timesteps
                        # todo change all_features to return only one segment
                        seq_current = seq[
                                      offset: offset + self.timesteps + self.timesteps,
                                      :].copy()
                        seq_current[
                            time_index - offset, voice_index] = pitch_index

                        left_features_current, _, _, _ = all_features(
                            seq_current,
                            voice_index=voice_index,
                            time_index=self.timesteps,
                            timesteps=self.timesteps,
                            num_pitches=num_pitches,
                            num_voices=num_voices)

                        # pad beginning of sequence with -1
                        left_features_current[:a - offset, :] = -1
                        input_features_current = {
                            'input_seq': left_features_current,
                        }

                        batch_current.append(input_features_current)

                    # convert input_features
                    batch_current = {
                        key: np.array(
                            [input_features[key] for input_features in
                             batch_current])
                        for key in batch_current[0].keys()
                    }

                    # create target
                    # todo use argument constraint['transpose']
                    if 'second_interval' in constraint:
                        raise NotImplementedError
                    elif 'target' in constraint:
                        batch_target = constraint['target']
                    else:
                        raise NotImplementedError

                    # predict all hidden_repr in parallel
                    hidden_reprs_current = self.hidden_repr_model.predict(
                        batch_current, batch_size=len(batch_current))
                    hidden_repr_target = \
                        self.hidden_repr_model.predict(batch_target,
                                                       batch_size=len(
                                                           batch_target))[0]
                    # predict all distances
                    dists = np.array(list(map(
                        lambda hidden_repr_current: spearman_rho(
                            hidden_repr_current, hidden_repr_target),
                        hidden_reprs_current)))

                    batched_dists.append(dists)
                    is_dist.append(True)

                    # print(np.min(dists), np.max(dists))
                    break
            if not in_constraint:
                batched_dists.append(no_dist)
                is_dist.append(False)
        return np.array(batched_dists), np.array(is_dist)

    def compute_stats(self, chorale_index=0, num_elements=1000, timesteps=32,
                      effective_timesteps=32):
        """

        :param num_elements:
        :return: features produced by generator that achieve minimum distance
        """
        X_pc, pc_index2pc, pc2pc_index, note_index2pc_index = pickle.load(
            open(self.shape_dataset_filepath, 'rb'))
        num_pitches = [len(pc_index2pc)]

        # init:
        # chorale_transpositions = X_pc[chorale_index]
        # sequence_length = len(chorale_transpositions[0][SEQ][SOP_INDEX])
        if effective_timesteps:
            generator_unitary = self.generator(batch_size=1, phase='all',
                                               effective_timestep=effective_timesteps)
        else:
            generator_unitary = self.generator_unitary
        distance_diff = []
        distance_same = []

        for _ in tqdm(range(num_elements)):
            hidden_repr_1 = self.hidden_repr_model.predict(
                next(generator_unitary)[0]['input_seq'])[0]
            hidden_repr_2 = self.hidden_repr_model.predict(
                next(generator_unitary)[0]['input_seq'])[0]

            distance_diff.append(spearman_rho(hidden_repr_1, hidden_repr_2))

            chorale_transpositions = np.random.choice(X_pc)

            chorale_length = len(chorale_transpositions[0][SEQ][SOP_INDEX])
            time_index = np.random.choice(chorale_length - timesteps)
            transposition_index_1, transposition_index_2 = np.random.choice(
                len(chorale_transpositions), size=2)
            transposition_1 = chorale_transpositions[transposition_index_1]
            transposition_2 = chorale_transpositions[transposition_index_2]
            hidden_repr_1 = self.hidden_repr_model.predict({
                'input_seq': chorale_to_onehot(transposition_1[SEQ][:,
                                               time_index: time_index + timesteps],
                                               num_pitches=num_pitches,
                                               time_major=False)[None, :, :]
            })[0]
            hidden_repr_2 = self.hidden_repr_model.predict({
                'input_seq': chorale_to_onehot(transposition_2[SEQ][:,
                                               time_index: time_index + timesteps],
                                               num_pitches=num_pitches,
                                               time_major=False)[None, :, :]
            })[0]

            distance_same.append(spearman_rho(hidden_repr_1, hidden_repr_2))

        hist_data = [distance_diff, distance_same]

        from plotly.offline import plot
        fig = ff.create_distplot(hist_data=hist_data,
                                 group_labels=['Random', 'Transposition'])

        plot(fig, filename='stats.html')

    def show_mean_distance_matrix(self, chorale_index=0, timesteps=32,
                                  show_plotly=False):
        """

        :param num_elements:
        :param features:
        :return: features produced by generator that achieve minimum distance
        """
        X_pc, pc_index2pc, pc2pc_index, note_index2pc_index = pickle.load(
            open(self.shape_dataset_filepath, 'rb'))
        num_pitches = [len(pc_index2pc)]

        # input:
        chorale_transpositions = X_pc[chorale_index]
        sequence_length = len(chorale_transpositions[0][SEQ][SOP_INDEX])
        num_chunks = sequence_length - timesteps

        mat = np.zeros((num_chunks, len(chorale_transpositions),
                        len(chorale_transpositions)))

        for time_index in tqdm(range(num_chunks)):
            for index_transposition_1, chorale_1 in enumerate(
                    chorale_transpositions):

                hidden_repr_1 = self.hidden_repr_model.predict({
                    'input_seq': chorale_to_onehot(
                        chorale_1[SEQ][:, time_index: time_index + timesteps],
                        num_pitches=num_pitches,
                        time_major=False)[None, :, :]
                })[0]
                for index_transposition_2, chorale_2 in enumerate(
                        chorale_transpositions):
                    hidden_repr_2 = self.hidden_repr_model.predict({
                        'input_seq': chorale_to_onehot(chorale_2[SEQ][:,
                                                       time_index: time_index + timesteps],
                                                       num_pitches=num_pitches,
                                                       time_major=False)[None,
                                     :, :]
                    })[0]
                    mat[
                        time_index, index_transposition_1, index_transposition_2] = spearman_rho(
                        hidden_repr_1,
                        hidden_repr_2)

        mean = np.mean(mat, axis=0)
        std = np.std(mat, axis=0)
        if show_plotly:
            from plotly.offline import plot
            from plotly import tools
            fig = tools.make_subplots(rows=1, cols=2,
                                      subplot_titles=('Mean', 'Std'))
            fig.append_trace(go.Heatmap(
                z=mean
            ), 1, 1)
            fig.append_trace(
                go.Heatmap(
                    z=std
                ), 1, 2)
            plot(fig, filename='mean_distance_matrix.html')

        else:
            print(mean)
            print(std)


class GenerationModel(SequentialModel):
    def __init__(self, name: str, dataset_name: str, timesteps=16, **kwargs):
        self.kwargs = kwargs
        if 'metadatas' in kwargs:
            self.metadatas = kwargs['metadatas']
        else:
            self.metadatas = []
        super(GenerationModel, self).__init__(name=name,
                                              model_type='generation',
                                              dataset_name=dataset_name,
                                              timesteps=timesteps)

    def create_dataset(self):
        raise NotImplementedError
        # Create pickled dataset
        if not os.path.exists(self.dataset_filepath):
            initialization(None,
                           metadatas=self.metadatas,
                           voice_ids=[SOP_INDEX],
                           BACH_DATASET=self.dataset_filepath)

    def generator(self, batch_size, phase, percentage_train=0.8,
                  voice_index=SOP_INDEX, **kwargs):
        X, voice_ids, index2notes, note2indexes, metadatas = pickle.load(
            open(self.dataset_filepath, 'rb'))
        num_pitches = list(map(lambda x: len(x), index2notes))
        num_voices = len(voice_ids)
        assert num_voices == 1
        # Set chorale_indices
        if phase == 'train':
            chorale_indices = np.arange(int(len(X) * percentage_train))
        elif phase == 'test':
            chorale_indices = np.arange(int(len(X) * percentage_train), len(X))
        elif phase == 'all':
            chorale_indices = np.arange(int(len(X)))
        else:
            NotImplementedError

        left_features = []
        right_features = []
        central_features = []
        left_metas = []
        right_metas = []
        central_metas = []
        labels = []

        batch = 0

        while True:
            # draw random element
            chorale_index = np.random.choice(chorale_indices)
            chorales = X[chorale_index]
            transposition_index_1 = np.random.randint(len(chorales))

            # chorale is voice major
            chorale, chorale_metadatas, offset_1 = chorales[
                transposition_index_1]
            # make chorale time major
            chorale = np.transpose(chorale)
            # draw time index
            chorale_length = len(chorale)
            time_index = np.random.randint(self.timesteps,
                                           chorale_length - self.timesteps)

            # pad with start and end symbols
            padding_dimensions = (self.timesteps, num_voices)
            start_symbols = np.array(list(
                map(lambda note2index: note2index[START_SYMBOL],
                    note2indexes)))
            end_symbols = np.array(list(
                map(lambda note2index: note2index[END_SYMBOL], note2indexes)))

            extended_chorale = np.concatenate(
                (np.full(padding_dimensions, start_symbols),
                 chorale,
                 np.full(padding_dimensions, end_symbols)),
                axis=0)
            extended_chorale_metadatas = [
                np.concatenate((np.zeros(self.timesteps),
                                metadata,
                                np.zeros(self.timesteps)),
                               axis=0) for metadata in chorale_metadatas]

            features = all_features(chorale=extended_chorale,
                                    voice_index=voice_index,
                                    time_index=time_index,
                                    timesteps=self.timesteps,
                                    num_pitches=num_pitches,
                                    num_voices=num_voices)
            left_meta, meta, right_meta = all_metadatas(
                chorale_metadatas=extended_chorale_metadatas,
                metadatas=metadatas,
                time_index=time_index, timesteps=self.timesteps)

            (left_feature, central_feature, right_feature,
             label
             ) = features

            left_features.append(left_feature)
            right_features.append(right_feature)
            central_features.append(central_feature)

            left_metas.append(left_meta)
            right_metas.append(right_meta)
            central_metas.append(meta)
            labels.append(label)

            batch += 1

            # if there is a full batch
            if batch == batch_size:
                # todo add central features if multiple voices
                next_element = (
                    {'left_features': np.array(left_features,
                                               dtype=np.float32),
                     'right_features':
                         np.array(right_features, dtype=np.float32),
                     'central_features':
                         np.array(central_features, dtype=np.float32),
                     'left_metas':
                         np.array(left_metas, dtype=np.float32),
                     'right_metas':
                         np.array(right_metas, dtype=np.float32),
                     'central_metas':
                         np.array(central_metas, dtype=np.float32),
                     }
                    ,
                    {'predictions': np.array(labels, dtype=np.float32)}
                )
                yield next_element

                batch = 0
                left_features = []
                central_features = []
                right_features = []
                left_metas = []
                right_metas = []
                central_metas = []
                labels = []

    def create_model(self):
        if self.name == 'skip':
            X, voice_ids, index2notes, note2indexes, metadatas = self.dataset
            num_pitches = list(map(len, index2notes))
            inputs, labels = next(self.generator_unitary)
            num_features_lr = inputs['left_features'].shape[-1]
            num_features_meta = inputs['left_metas'].shape[-1]
            num_pitches = num_pitches[SOP_INDEX]
            return deepbach_skip_sop(num_features_lr=num_features_lr,
                                     num_features_meta=num_features_meta,
                                     num_pitches=num_pitches,
                                     num_units_lstm=[128, 128],
                                     num_dense=128,
                                     timesteps=self.timesteps)
        else:
            raise NotImplementedError

    def gibbs(self, chorale_metas=None,
              sequence_length=50, num_iterations=1000, temperature=1.,
              batch_size_per_voice=16,
              show_results=False):
        """

        :param chorale_metas:
        :param sequence_length:
        :param num_iterations:
        :param temperature:
        :param batch_size_per_voice:
        :param show_results:
        :return: voice major numpy array
        """

        X, X_metadatas, voices_ids, index2notes, note2indexes, metadatas = self.dataset
        num_pitches = list(map(len, index2notes))
        num_voices = len(voices_ids)

        # set sequence_length
        if chorale_metas is not None:
            # do not generate a sequence longer than chorale_metas if not None
            sequence_length = len(chorale_metas[0])

        seq = np.zeros(
            shape=(2 * self.timesteps + sequence_length, num_voices))

        # init seq
        for expert_index in range(num_voices):
            # Add start and end symbol + random init
            seq[:self.timesteps, expert_index] = [note2indexes[expert_index][
                                                      START_SYMBOL]] * self.timesteps
            seq[self.timesteps:-self.timesteps,
            expert_index] = np.random.randint(num_pitches[expert_index],
                                              size=sequence_length)

            seq[-self.timesteps:, expert_index] = [note2indexes[expert_index][
                                                       END_SYMBOL]] * self.timesteps

        # extend chorale_metas
        if chorale_metas is not None:
            # chorale_metas is a list
            extended_chorale_metas = [
                np.concatenate((np.zeros((self.timesteps,)),
                                chorale_meta,
                                np.zeros((self.timesteps,))),
                               axis=0)
                for chorale_meta in chorale_metas]

        else:
            extended_chorale_metas = [
                np.concatenate((np.zeros((self.timesteps,)),
                                metadata.generate(sequence_length),
                                np.zeros((self.timesteps,))),
                               axis=0)
                for metadata in self.metadatas]

        min_temperature = temperature
        temperature = 1.5

        # todo generalize to multiple voices
        min_voice = SOP_INDEX
        voice_index = SOP_INDEX

        # Main loop
        for iteration in tqdm(range(num_iterations)):

            # Simulated annealing
            temperature = max(min_temperature, temperature * 0.99)

            # prediction
            probas, time_indexes = self.gibbs_step(seq=seq,
                                                   chorale_metas=extended_chorale_metas,
                                                   batch_size_per_voice=batch_size_per_voice,
                                                   voice_index=voice_index,
                                                   num_pitches=num_pitches,
                                                   sequence_length=sequence_length)

            # update
            for batch_index, proba in enumerate(probas):
                # use temperature
                probas_pitch = np.log(proba) / temperature
                probas_pitch = np.exp(probas_pitch) / np.sum(
                    np.exp(probas_pitch)) - 1e-7

                # pitch can include slur_symbol
                pitch = np.argmax(np.random.multinomial(1, probas_pitch))
                seq[time_indexes[batch_index], voice_index] = pitch

        seq = np.transpose(np.array(seq[self.timesteps:-self.timesteps, :]))

        # show results in editor
        if show_results:
            # convert
            score = indexed_chorale_to_score(seq,
                                             pickled_dataset=self.dataset_filepath
                                             )
            score.show()

        return seq

    def gibbs_step(self, seq, chorale_metas, batch_size_per_voice,
                   voice_index,
                   num_pitches, max_timestep=None):
        """
        This is a batched Gibbs step
        :param voice_index:
        :param batch_size_per_voice:
        :param chorale_metas:
        :param num_pitches:
        :param seq: padded with max_timesteps begin and end symbols
        :return:
        """
        if not max_timestep:
            max_timestep = self.timesteps

        batch_input_features = []
        time_indexes = []

        for batch_index in range(batch_size_per_voice):
            # choose time_index
            time_index = np.random.randint(max_timestep,
                                           len(seq) - max_timestep)
            time_indexes.append(time_index)

            (left_feature,
             central_feature,
             right_feature,
             label) = all_features(seq, voice_index, time_index,
                                   self.timesteps, num_pitches,
                                   self.num_voices)

            left_metas, central_metas, right_metas = all_metadatas(
                chorale_metadatas=chorale_metas,
                metadatas=self.metadatas,
                time_index=time_index, timesteps=self.timesteps)

            input_features = {'left_features': left_feature[:, :],
                              # 'central_features': central_feature[:],
                              'right_features': right_feature[:, :],
                              'left_metas': left_metas,
                              'central_metas': central_metas,
                              'right_metas': right_metas}

            # list of dicts: predict need dict of numpy arrays
            batch_input_features.append(input_features)

        # convert input_features
        batch_input_features = {key: np.array(
            [input_features[key] for input_features in batch_input_features])
            for key in batch_input_features[0].keys()
        }
        # make all estimations
        probas = self.model.predict(batch_input_features,
                                    batch_size=batch_size_per_voice)
        return probas, time_indexes


class VariationsModel:
    def __init__(self, generation_model_name, distance_model_name,
                 dataset_name,
                 timesteps_generation=16, timesteps_distance=32,
                 is_shape_model=False,
                 distance_model_kwargs={}):
        self.is_shape_model = is_shape_model
        self.distance_model_name = distance_model_name
        self.generation_model_name = generation_model_name
        self.generation_model = GenerationModel(name=generation_model_name,
                                                dataset_name=dataset_name,
                                                timesteps=timesteps_generation)
        # self.distance_model = DistanceModel(name=distance_model_name,
        #                                     dataset_name=dataset_name,
        #                                     timesteps=timesteps_distance)
        if is_shape_model:
            self.distance_model = ShapeModel(name=distance_model_name,
                                             dataset_name=dataset_name,
                                             timesteps=timesteps_distance)
        else:
            self.distance_model = InvariantDistanceModel(
                name=distance_model_name,
                dataset_name=dataset_name,
                timesteps=timesteps_distance,
                **distance_model_kwargs)
        self.dataset = self.generation_model.dataset
        self.dataset_filepath = self.generation_model.dataset_filepath
        self.metadatas = self.generation_model.metadatas

    def gibbs(self, chorale_metas=None,
              sequence_length=50, num_iterations=1000, temperature=1.,
              batch_size_per_voice=16,
              list_of_constraints=None,
              show_results=False):
        """

        :param chorale_metas:
        :param sequence_length:
        :param num_iterations:
        :param temperature:
        :param batch_size_per_voice:
        :param show_results:
        :return: voice major numpy array
        """
        X, voices_ids, index2notes, note2indexes, metadatas = self.dataset
        num_pitches = list(map(len, index2notes))
        num_voices = len(voices_ids)
        max_timestep = max(self.generation_model.timesteps,
                           self.distance_model.timesteps)

        # set sequence_length
        if chorale_metas is not None:
            # do not generate a sequence longer than chorale_metas if not None
            sequence_length = len(chorale_metas[0])

        seq = np.zeros(shape=(2 * max_timestep + sequence_length, num_voices))

        # init seq
        for expert_index in range(num_voices):
            # Add start and end symbol + random init
            seq[:max_timestep, expert_index] = [note2indexes[expert_index][
                                                    START_SYMBOL]] * max_timestep
            seq[max_timestep:-max_timestep, expert_index] = np.random.randint(
                num_pitches[expert_index],
                size=sequence_length)

            seq[-max_timestep:, expert_index] = [note2indexes[expert_index][
                                                     END_SYMBOL]] * max_timestep

        # extend chorale_metas
        if chorale_metas is not None:
            # chorale_metas is a list
            extended_chorale_metas = [
                np.concatenate((np.zeros((max_timestep,)),
                                chorale_meta,
                                np.zeros((max_timestep,))),
                               axis=0) for chorale_meta in chorale_metas]

        else:
            extended_chorale_metas = [
                np.concatenate((np.zeros((max_timestep,)),
                                metadata.generate(sequence_length),
                                np.zeros((max_timestep,))),
                               axis=0) for metadata in self.metadatas]

        # add offsets in list_of_constraints
        for constraint in list_of_constraints:
            a, b = constraint['first_interval']
            constraint['first_interval'] = (a + max_timestep, b + max_timestep)
            if 'second_interval' in constraint:
                a, b = constraint['second_interval']
                constraint['second_interval'] = (
                    a + max_timestep, b + max_timestep)

        min_temperature = temperature
        min_temperature = 1.
        temperature = 3.

        distance_temperature = 5.
        min_distance_temperature = 2.

        max_std_dist = 3.
        std_dist = 0.5

        discount_factor_temperature = np.power(min_temperature / temperature,
                                               (3 / 2) / num_iterations)
        discount_factor_distance_temperature = np.power(
            min_distance_temperature / distance_temperature,
            (4 / 3) / num_iterations)

        # todo generalize to multiple voices
        min_voice = SOP_INDEX
        voice_index = SOP_INDEX

        # define uniform prob
        uniform = np.ones((num_pitches[voice_index],))
        uniform /= np.sum(uniform)

        # Main loop
        for iteration in tqdm(range(num_iterations)):
            # Simulated annealing
            temperature = max(min_temperature,
                              temperature * discount_factor_temperature)
            print('temperature:' + str(temperature))
            # std_dist = min(max_std_dist, std_dist * 1.05)
            # print('std_dist:' + str(std_dist))
            distance_temperature = max(min_distance_temperature,
                                       distance_temperature * discount_factor_distance_temperature)
            print('distance_temperature:' + str(distance_temperature))

            # prediction
            probas_generation, time_indexes = self.generation_model.gibbs_step(
                seq=seq,
                chorale_metas=extended_chorale_metas,
                batch_size_per_voice=batch_size_per_voice,
                voice_index=voice_index,
                num_pitches=num_pitches,
                max_timestep=max_timestep
            )
            # distance biases
            if self.is_shape_model:
                # convert seq to seq of pcs
                assert num_voices == 1
                assert isinstance(self.distance_model, ShapeModel)
                num_pcs = [len(self.distance_model.pc_index2pc)]
                seq_pcs = np.array(list(
                    map(lambda note: self.distance_model.note_index2pc_index,
                        seq[:, SOP_INDEX])))[:, None]

                batched_dists_pc, is_dist = self.distance_model.batched_distances(
                    seq=seq_pcs,
                    voice_index=voice_index,
                    time_indexes=time_indexes,
                    list_of_constraints=list_of_constraints,
                    num_pitches=num_pcs
                )
                # transforms dists on pc to dists on notes

                batched_dists = np.zeros(
                    (len(time_indexes), len(index2notes[SOP_INDEX])))
                for pc_index, note_index in self.distance_model.pc_index2note_indexes.items():
                    for batch_index, _ in enumerate(time_indexes):
                        batched_dists[batch_index, note_index] = \
                            batched_dists_pc[batch_index, pc_index]
            else:
                batched_dists, is_dist = self.distance_model.batched_distances(
                    seq=seq,
                    voice_index=voice_index,
                    time_indexes=time_indexes,
                    list_of_constraints=list_of_constraints,
                    num_pitches=num_pitches
                )

            # if no constraint
            # todo can be removed?
            # if list_of_constraints is None:
            #     for batch_index in range(batch_size_per_voice):
            #         batched_dists.append(uniform)
            #     return np.array(batched_dists)


            # update
            for batch_index, (p_gen, dist, consider_dist) in enumerate(
                    zip(probas_generation, batched_dists, is_dist)):

                # tweak probas
                if consider_dist:
                    # todo divide by std?
                    # p_dist = (dist - np.mean(dist)) / np.std(dist)
                    # use temperature on distance
                    p_dist = (dist - np.mean(dist)) / distance_temperature

                    p_dist = np.exp(-p_dist)
                    p_dist = np.exp(p_dist) / (np.sum(np.exp(p_dist)) + 1e-6)

                    # print('p_dist:')
                    # print(min(p_dist), max(p_dist))
                    # print('std:' + str(np.std(dist)))

                    print('p_dist_temp:')
                    print(min(p_dist), max(p_dist))

                    print('p_gen:')
                    print(min(p_gen), max(p_gen))
                    # use temperature on p_gen
                    p_gen = np.log(p_gen) / temperature
                    p_gen = np.exp(p_gen) / (np.sum(np.exp(p_gen)) + 1e-6)
                    print('p_gen_temp:')
                    print(min(p_gen), max(p_gen))

                    # todo multiply and renormalize?
                    probas_pitch = p_gen * p_dist
                    probas_pitch = probas_pitch / np.sum(probas_pitch)
                    # todo sum?
                else:
                    # use temperature on p_gen
                    probas_pitch = p_gen
                    probas_pitch = np.log(probas_pitch) / temperature
                    probas_pitch = np.exp(probas_pitch) / (
                        np.sum(np.exp(probas_pitch)) + 1e-6)
                # use temperature on generation

                # pitch can include slur_symbol
                pitch = np.argmax(np.random.multinomial(1, probas_pitch))
                seq[time_indexes[batch_index], voice_index] = pitch

                # ---------PRINT DISTANCE TO REMOVE---------------
                # a, b = list_of_constraints[0]['first_interval']
                # offset = b - self.distance_model.timesteps
                # seq_current = seq[offset: offset + self.distance_model.timesteps + self.distance_model.timesteps, :].copy()
                #
                # left_features_current, _, _, _ = all_features(seq_current,
                #                                               voice_index=voice_index,
                #                                               time_index=self.distance_model.timesteps,
                #                                               timesteps=self.distance_model.timesteps,
                #                                               num_pitches=num_pitches,
                #                                               num_voices=num_voices)
                #
                # # pad beginning of sequence with -1
                # left_features_current[:a - offset, :] = -1
                # input_features_current = {'input_seq': np.array([left_features_current]),
                #                           }
                #
                # # create target
                # #
                # # a, b = list_of_constraints[0]['second_interval']
                # # offset = b - self.distance_model.timesteps
                # # # todo change all_features to return only one segment
                # # seq_target = seq[offset: offset + self.distance_model.timesteps + self.distance_model.timesteps,
                # #              :]  # .copy()
                # #
                # # left_features_target, _, _, _ = all_features(seq_target,
                # #                                              voice_index=voice_index,
                # #                                              time_index=self.distance_model.timesteps,
                # #                                              timesteps=self.distance_model.timesteps,
                # #                                              num_pitches=num_pitches,
                # #                                              num_voices=num_voices)
                # # # pad beginning of sequence with -1
                # # left_features_target[: a - offset, :] = -1
                # #
                # # input_features_target = {'input_seq': np.array([left_features_target]),
                # #                          }
                #
                # input_features_target = list_of_constraints[0]['target']
                #
                # # predict all hidden_repr in parallel
                # hidden_repr_current = self.distance_model.hidden_repr_model.predict(input_features_current, batch_size=1)[0]
                # hidden_repr_target = self.distance_model.hidden_repr_model.predict(input_features_target, batch_size=1)[0]
                # # predict all distances
                # dist = spearman_rho(hidden_repr_current, hidden_repr_target)
                # print('Dist: ' + str(dist))
                # ---------PRINT DISTANCE TO REMOVE---------------

        seq = np.transpose(np.array(seq[max_timestep:-max_timestep, :]))

        # show results in editor
        if show_results:
            # convert
            score = indexed_chorale_to_score(seq,
                                             pickled_dataset=self.dataset_filepath)
        score.show()

        return seq


def test_masking():
    timesteps_distance = 32
    distance_model = DistanceModel(name='seq2seq_masking',
                                   dataset_name='bach_sop',
                                   timesteps=timesteps_distance,
                                   dropout=0.2)
    if train:
        distance_model.train(batch_size=batch_size, nb_epochs=nb_epochs,
                             steps_per_epoch=samples_per_epoch,
                             validation_steps=nb_val_samples)

    batch = next(distance_model.generator_unitary)
    target = batch[0]['input_seq'][0]
    prediction = distance_model.model.predict(batch[0], batch_size=1)[0]

    score = indexed_chorale_to_score(np.array([list(map(np.argmax, target))]),
                                     pickled_dataset=distance_model.dataset_filepath
                                     )
    score.show()

    score = indexed_chorale_to_score(
        np.array([list(map(np.argmax, prediction))]),
        pickled_dataset=distance_model.dataset_filepath
    )
    score.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps',
                        help="model's range (default: %(default)s)",
                        type=int, default=16)
    parser.add_argument('-b', '--batch_size_train',
                        help='batch size used during training phase (default: %(default)s)',
                        type=int, default=128)
    parser.add_argument('-s', '--samples_per_epoch',
                        help='number of samples per epoch (default: %(default)s)',
                        type=int, default=12800 * 7)
    parser.add_argument('--num_val_samples',
                        help='number of validation samples (default: %(default)s)',
                        type=int, default=2560)
    parser.add_argument('-u', '--num_units_lstm', nargs='+',
                        help='number of lstm units (default: %(default)s)',
                        type=int, default=[200, 200])
    parser.add_argument('-d', '--num_dense',
                        help='size of non recurrent hidden layers (default: %(default)s)',
                        type=int, default=200)
    parser.add_argument('-n', '--name',
                        help='model name (default: %(default)s)',
                        choices=['deepbach', 'skip', 'norelu'],
                        type=str, default='skip')
    parser.add_argument('-i', '--num_iterations',
                        help='number of gibbs iterations (default: %(default)s)',
                        type=int, default=20000)
    parser.add_argument('-t', '--train', nargs='?',
                        help='train models for N epochs (default: 15)',
                        default=0, const=15, type=int)
    parser.add_argument('-p', '--parallel', nargs='?',
                        help='number of parallel updates (default: 16)',
                        type=int, const=16, default=1)
    parser.add_argument('--overwrite',
                        help='overwrite previously computed models',
                        action='store_true')
    parser.add_argument('-m', '--midi_file', nargs='?',
                        help='relative path to midi file',
                        type=str, const='datasets/god_save_the_queen.mid')
    parser.add_argument('-l', '--length',
                        help='length of unconstrained generation',
                        type=int, default=160)
    parser.add_argument('--ext',
                        help='extension of model name',
                        type=str, default='')
    parser.add_argument('-o', '--output_file', nargs='?',
                        help='path to output file',
                        type=str, default='',
                        const='generated_examples/example.mid')
    parser.add_argument('--dataset', nargs='?',
                        help='path to dataset folder',
                        type=str, default='')
    parser.add_argument('-r', '--reharmonization', nargs='?',
                        help='reharmonization of a melody from the corpus identified by its id',
                        type=int)
    args = parser.parse_args()
    print(args)

    if args.ext:
        ext = '_' + args.ext
    else:
        ext = ''

    # metadatas = [TickMetadatas(SUBDIVISION), FermataMetadatas(), KeyMetadatas(window_size=1)]
    metadatas = [TickMetadatas(SUBDIVISION), FermataMetadatas()]

    timesteps_generation = args.timesteps
    batch_size = args.batch_size_train
    samples_per_epoch = args.samples_per_epoch
    nb_val_samples = args.num_val_samples
    num_units_lstm = args.num_units_lstm
    model_name = args.name.lower()
    sequence_length = args.length
    batch_size_per_voice = args.parallel
    num_units_lstm = args.num_units_lstm
    num_dense = args.num_dense
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = None

    parallel = batch_size_per_voice > 1
    train = args.train > 0
    nb_epochs = args.train
    overwrite = args.overwrite

    generation_model_name = 'skip'

    dataset_name = 'transpose/bach_sop'
    # Generation model
    # generation_model = GenerationModel(name=generation_model_name, dataset_name=dataset_name,
    #                                    timesteps=timesteps_generation,
    #                                    metadatas=metadatas)
    # if train:
    #     generation_model.train(batch_size=batch_size, nb_epochs=nb_epochs, samples_per_epoch=samples_per_epoch,
    #                            nb_val_samples=nb_val_samples)


    # generation_model.gibbs(sequence_length=16 * 16,
    #                        num_iterations=1000,
    #                        show_results=True)

    # ----------DISTANCE MODEL----------
    timesteps_distance = 32
    # distance_model_name = 'seq2seq'
    # distance_model = DistanceModel(name=distance_model_name,
    #                                dataset_name=dataset_name,
    #                                timesteps=timesteps_distance,
    #                                dropout=0.2)
    #
    # # distance_model.find_nearests(next(distance_model.generator(batch_size=1,
    # #                                                            phase='all',
    # #                                                            effective_timestep=32))[0],
    # #                              show_results=True,
    # #                              effective_timestep=32,
    # #                              num_elements=10000)
    # if train:
    #     # distance_model.train(batch_size=batch_size, nb_epochs=nb_epochs, samples_per_epoch=samples_per_epoch,
    #     #                      nb_val_samples=nb_val_samples)
    #     distance_model.train(batch_size=128,
    #                          nb_epochs=50,
    #                          samples_per_epoch=1024 * 20,
    #                          nb_val_samples=1024 * 2,
    #                          overwrite=True,
    #                          effective_timestep=32,
    #                          percentage_train=0.9)
    #
    # distance_model.compute_stats()
    # exit()

    # INVARIANT DISTANCE
    # invariant_distance_model_name = 'seq2seq_large'
    # invariant_distance_model_name = 'seq2seq_absolute_invariant'
    # invariant_distance_model_name = 'seq2seq_absolute_invariant_reg'
    # invariant_distance_model_name = 'seq2seq_absolute_invariant_reg_mean'
    invariant_distance_model_name = 'seq2seq_absolute_invariant_reg_mean_relu'
    distance_model_kwargs = dict(
        reg='l2',
        # reg=None,
        # dropout_prob=0.2
        dropout_prob=None,
        lambda_reg=1e-5,
        num_layers=2,
        num_units_lstm=512,
    )

    invariant_distance_model = InvariantDistanceModel(
        name=invariant_distance_model_name,
        dataset_name=dataset_name,
        timesteps=timesteps_distance,
        **distance_model_kwargs
    )
    if train:
        invariant_distance_model.train(batch_size=256,
                                       nb_epochs=98,
                                       steps_per_epoch=128,
                                       validation_steps=8,
                                       overwrite=True,
                                       effective_timestep=32,
                                       percentage_train=0.9)

    # invariant_distance_model.find_nearests(
    #     next(invariant_distance_model.generator(batch_size=1,
    #                                             phase='all',
    #                                             effective_timestep=32))[0],
    #     show_results=True,
    #     effective_timestep=32,
    #     num_elements=1000)
    # invariant_distance_model.show_preds(effective_timestep=32)
    # invariant_distance_model.test_transpose_out_of_bounds(
    #     effective_timestep=32)
    invariant_distance_model.show_distance_matrix(chorale_index=241,
                                                  time_index=32,
                                                  show_plotly=True)
    # invariant_distance_model.show_mean_distance_matrix(chorale_index=0,
    #                                                    show_plotly=True)
    invariant_distance_model.compute_stats(chorale_index=0, num_elements=1000)
    # invariant_distance_model.show_all_absolute_preds(effective_timestep=32)
    exit()

    # ----------SHAPE MODEL-----------
    # timesteps_distance = 32
    # # shape_model_name = 'seq2seq_large'
    # shape_model_name = 'seq2seq_absolute_invariant_reg_l1'
    # shape_model = ShapeModel(name=shape_model_name,
    #                          dataset_name=dataset_name,
    #                          timesteps=timesteps_distance,
    #                          dropout=0.02)
    # if train:
    #     shape_model.train(batch_size=128,
    #                       nb_epochs=50,
    #                       samples_per_epoch=1024 * 20,
    #                       nb_val_samples=1024 * 2,
    #                       overwrite=True,
    #                       effective_timestep=32,
    #                       percentage_train=0.9)

    # shape_model.show_preds(effective_timestep=32)

    # shape_model.find_nearests(next(shape_model.generator(batch_size=1,
    #                                                      phase='all',
    #                                                      effective_timestep=32))[0],
    #                           show_results=True,
    #                           effective_timestep=32,
    #                           num_elements=1000)
    # shape_model.compute_stats(
    #     num_elements=10000,
    #     effective_timesteps=32)
    # shape_model.show_mean_distance_matrix(chorale_index=1, show_plotly=True)
    # exit()
    # --------VARIATION MODEL----------
    # VARIATION MODEL WITH SHAPE MODEL
    # X_pc, pc_index2pc, pc2pc_index, note_index2pc_index = pickle.load(open(shape_model.shape_dataset_filepath, 'rb'))

    # X[0][2][0] = indexed (1, time)
    # target_chorale = X_pc[2][2][0][:, 16: 16 + 32]
    # CREATE TARGET SHAPE



    # target must contain be a batch of size 1
    # target = {'input_seq': np.array([chorale_to_onehot(np.transpose(target_chorale), list(map(len, [pc_index2pc])))])}
    # # show target
    # score_nearest = indexed_seq_to_score(target_chorale[0], shape_model.pc_index2pc,
    #                                      shape_model.pc2pc_index)
    # score_nearest.show()
    #
    # list_of_constraints = [{'first_interval': (16 * 1, 16 * 3),
    #                         'target': target
    #                         },
    #                        {'first_interval': (16 * 4, 16 * 6),
    #                         'target': target
    #                         }
    #                        ]

    # list_of_constraints = [
    #     {'first_interval': (16 * 1, 16 * 3),
    #      'second_interval': (16 * 4, 16 * 6),
    #      'transpose': 0},
    #     {'first_interval': (16 * 4, 16 * 6),
    #      'second_interval': (16 * 1, 16 * 3),
    #      'transpose': 0}
    # ]


    # Variations model with shape model
    # variations_model = VariationsModel(generation_model_name=generation_model_name,
    #                                    distance_model_name=shape_model_name,
    #                                    dataset_name=dataset_name,
    #                                    timesteps_generation=timesteps_generation,
    #                                    timesteps_distance=timesteps_distance,
    #                                    is_shape_model=True
    #                                    )
    #
    # variations_model.gibbs(sequence_length=16 * 7,
    #                        num_iterations=500,
    #                        show_results=True,
    #                        batch_size_per_voice=4,
    #                        list_of_constraints=list_of_constraints)
    #
    # exit()


    # Variations model with invariant distance
    X_pc, voice_ids, pc_index2pc, note2indexes, metadatas = invariant_distance_model.dataset
    sequence_length = 16 * 7
    target_chorale = X_pc[2][2][0][:, 32: 32 + 32]
    chorale_metas = [meta[32: 32 + sequence_length] for meta in
                     X_pc[2][2][META]]

    # CREATE TARGET SHAPE
    # target must contain be a batch of size 1
    target = {'input_seq': np.array([chorale_to_onehot(
        np.transpose(target_chorale), list(map(len, pc_index2pc)))])}
    # show target
    score_nearest = indexed_seq_to_score(target_chorale[0],
                                         invariant_distance_model.index2notes[
                                             SOP_INDEX],
                                         invariant_distance_model.note2indexes[
                                             SOP_INDEX])
    score_nearest.show()

    list_of_constraints = [{'first_interval': (16 * 1, 16 * 3),
                            'target': target
                            },
                           {'first_interval': (16 * 4, 16 * 6),
                            'target': target
                            }
                           ]

    variations_model = VariationsModel(
        generation_model_name=generation_model_name,
        distance_model_name=invariant_distance_model_name,
        dataset_name=dataset_name,
        timesteps_generation=timesteps_generation,
        timesteps_distance=timesteps_distance,
        is_shape_model=False,
        distance_model_kwargs=distance_model_kwargs
    )

    variations_model.gibbs(sequence_length=16 * 7,
                           chorale_metas=chorale_metas,
                           num_iterations=500,
                           show_results=True,
                           batch_size_per_voice=4,
                           list_of_constraints=list_of_constraints)
