import collections
import os
import pickle

import numpy as np
import torch
from deepPermutations.data_preprocessing import SOP_INDEX
from deepPermutations.data_utils import START_SYMBOL, END_SYMBOL, to_onehot, \
    first_note_index
from torch import nn


class SequentialModel(nn.Module):
    """
    Base class for distance models
    """

    def __init__(self, model_type: str, dataset_name: str,
                 timesteps=16, **kwargs):
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
        self.filepath = f'models/{self.model_type}_{params_string}.h5'

    def __str__(self):
        return self.filepath

    def hidden_repr(self, *input):
        NotImplementedError

    def forward(self, *input):
        NotImplementedError

    def generator(self, batch_size, phase, percentage_train=0.8, **kwargs):
        raise NotImplementedError

    def loss_and_acc_on_epoch(self, batches_per_epoch, generator, train=True,
                              reg_mat=None, reg_norm='l1'):

        mean_mce_loss = 0
        mean_accuracy = 0
        mean_reg = 0
        if train:
            self.train()
        else:
            self.eval()
        for sample_id, next_element in tqdm(
                enumerate(islice(generator, batches_per_epoch))):
            mce_loss, grad_reg, acc = self.loss_and_acc(reg_mat,
                                                        next_element,
                                                        reg_norm,
                                                        train=train)

            mean_mce_loss += mce_loss
            mean_reg += grad_reg
            mean_accuracy += acc

        return (mean_mce_loss / batches_per_epoch,
                mean_reg / batches_per_epoch,
                mean_accuracy / batches_per_epoch)

    def loss_and_acc(self, reg_mat, next_element, reg_norm, train):
        input_seq, input_seq_index = next_element

        g = Variable(reg_mat.cuda())

        # to cuda
        input_seq, input_seq_index = (
            Variable(torch.LongTensor(input_seq).cuda()),
            Variable(torch.LongTensor(input_seq_index).cuda())
        )

        optimizer.zero_grad()

        # forward pass
        weights, softmax, z = self.forward(x=input_seq, g=g)

        # mce_loss
        mce_loss = crossentropy_loss(weights, input_seq_index)

        # compute loss
        loss = mce_loss

        # regularization
        if reg_norm is not None:
            grad_reg = self.grad_reg(g, input_seq.size()[1], reg_norm, z)
            loss += self.lambda_reg * grad_reg

            # z regularization
            # l2 norm on z:
            z_reg = torch.pow(z, 2).sum(1).mean()
            # l2 norm on z except for first dim
            # z_reg = z
            # z_reg[:, 0] = 0
            # z_reg = torch.pow(z_reg, 2).sum(1).mean()

            loss += z_reg

        # backward pass and step
        if train:
            loss.backward()
            optimizer.step()

        # accuracy
        acc = accuracy(weights, input_seq_index)

        # compute mean loss and accuracy
        return (variable2float(mce_loss),
                variable2float(grad_reg),
                acc)

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

    def train_model(self,
                    batch_size,
                    batches_per_epoch,
                    num_epochs,
                    sequence_length,
                    plot=False,
                    save_every=10,
                    reg_norm=None):
        generator_train = load_dataset(batch_size=batch_size,
                                       timesteps=sequence_length,
                                       phase='train')
        generator_val = load_dataset(batch_size=batch_size,
                                     timesteps=sequence_length,
                                     phase='test')

        reg_mat = load_regularization_mat(name='num_notes',
                                          batch_size=batch_size,
                                          timesteps=sequence_length)
        # only to get res_size
        res = self.loss_and_acc_on_epoch(
            batches_per_epoch=1,
            generator=generator_train,
            train=True,
            reg_mat=reg_mat,
            reg_norm=reg_norm)
        res_size = len(res)

        if plot:
            import matplotlib.pyplot as plt
            fig, axarr = plt.subplots(res_size, sharex=True)
            x = []
            ys = [[] for _ in range(res_size)]
            ys_val = [[] for _ in range(res_size)]
            fig.show()

        for epoch_index in range(num_epochs):
            # train
            res = self.loss_and_acc_on_epoch(
                batches_per_epoch=batches_per_epoch,
                generator=generator_train,
                train=True,
                reg_mat=reg_mat,
                reg_norm=reg_norm)

            # eval
            res_val = self.loss_and_acc_on_epoch(
                batches_per_epoch=batches_per_epoch // 10,
                generator=generator_val,
                train=False,
                reg_mat=reg_mat,
                reg_norm=reg_norm)

            # plot
            if plot:
                x.append(epoch_index)
                for res_index in range(res_size):
                    y = res[res_index]
                    ys[res_index].append(y)

                    y_val = res_val[res_index]
                    ys_val[res_index].append(y_val)

                    axarr[res_index].plot(x, ys[res_index], 'r-',
                                          x, ys_val[res_index], 'r--')
                fig.canvas.draw()
                plt.pause(0.001)

            print(f"{',  '.join(map(str, res_val))}\n")

            if (epoch_index + 1) % save_every == 0:
                self.save()
                print('Model saved')

    def save(self):
        torch.save(self.state_dict(), self.filepath)

    def load(self):
        state_dict = torch.load(self.filepath)
        self.load_state_dict(state_dict)

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


class InvariantDistance(SequentialModel):
    def __init__(self, dataset_name, timesteps, **kwargs):
        model_type = 'invariant_distance'
        super(InvariantDistance, self).__init__(model_type,
                                                dataset_name,
                                                timesteps,
                                                **kwargs)

        # Parameters

    def numpy_indexed2onehot_chunk(self, indexed_chorale,
                                   start_symbols,
                                   end_symbols,
                                   num_pitches,
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
            axis=0)

        # to onehot
        input_seq = np.array(
            list(map(lambda x: to_onehot(x, num_pitches), input_seq)))

        return input_seq[time_index: time_index + self.timesteps, :]

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

        input_offsets = []
        first_notes = []

        sequences = [[] for _ in range(3)]

        batch = 0

        while True:
            chorale_index = np.random.choice(chorale_indices)
            chorales = X[chorale_index]
            if len(chorales) == 1:
                continue
            if not effective_timestep:
                effective_timestep = np.random.randint(min(8, self.timesteps),
                                                       self.timesteps + 1)

            transposition_indexes = np.random.choice(len(chorales),
                                                     replace=True, size=3)

            for seq_index, seq_list in enumerate(sequences):
                transposition_index = transposition_indexes[seq_index]
                seq, _, offset = np.array(chorales[transposition_index])

                # padding of size timesteps
                chorale_length = len(seq) + 2 * self.timesteps
                time_index = np.random.randint(0,
                                               chorale_length - self.timesteps)
                onehot_chunk = self.numpy_indexed2onehot_chunk(seq,
                                                               start_symbols,
                                                               end_symbols,
                                                               num_pitches,
                                                               time_index)
                seq_list.append(onehot_chunk)

            # find first note symbol of output seq (last sequence in sequences)
            first_note_index_output_seq = first_note_index(seq[SOP_INDEX],
                                                           time_index_start=time_index + self.timesteps - effective_timestep,
                                                           time_index_end=time_index + self.timesteps,
                                                           note2index=
                                                           note2indexes[
                                                               SOP_INDEX])
            first_note_output_seq = to_onehot(first_note_index_output_seq,
                                              num_indexes=num_pitches)
            first_notes.append(first_note_output_seq)

            batch += 1
            # if there is a full batch
            if batch == batch_size:
                torch_sequences = [torch.from_numpy(np.array(chunk))
                                   for chunk in sequences]
                torch_first_notes = torch.from_numpy(np.array(first_notes))

                # inputs, output, first_note
                next_element = ((torch_sequences[0], torch_sequences[1]),
                                torch_sequences[2],
                                torch_first_notes
                                )
                yield next_element

                batch = 0
                first_notes = []
                sequences = [[] for _ in range(3)]
