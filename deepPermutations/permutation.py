import argparse
import os
import pickle
from itertools import islice

from DeepBach.model_manager import train_models, load_models
from DeepBach.metadata import *

from keras.engine import Model
from keras.models import load_model
from scipy.stats import spearmanr, rankdata
from tqdm import tqdm

from data_utils import initialization, indexed_chorale_to_score, \
    generator_from_raw_dataset, START_SYMBOL, \
    END_SYMBOL, all_metadatas, all_features, chorale_to_onehot

SOP_INDEX = 0
ALTO_INDEX = 1
TENOR_INDEX = 2
BASS_INDEX = 3

BACH_DATASET = 'datasets/bach_sop.pickle'
BACH_DATASET_NO_META = 'datasets/bach_sop_no_meta.pickle'
BACH_ONEHOT_DATASET = 'datasets/bach_sop_onehot.pickle'

SUBDIVISION = 4


def create_onehot_corpus(timesteps, pickled_indexes_dataset=BACH_DATASET,
                         pickled_onehot_dataset=BACH_ONEHOT_DATASET):
    X, X_metadatas, voice_ids, index2notes, note2indexes, metadatas = pickle.load(
        open(pickled_indexes_dataset, 'rb'))
    num_pitches = list(map(lambda x: len(x), index2notes))
    num_voices = len(voice_ids)

    X_onehot = []

    for chorale_index, chorale in tqdm(enumerate(X)):
        extended_chorale = np.transpose(chorale)
        padding_dimensions = (timesteps,) + extended_chorale.shape[1:]
        start_symbols = np.array(list(
            map(lambda note2index: note2index[START_SYMBOL], note2indexes)))
        end_symbols = np.array(
            list(map(lambda note2index: note2index[END_SYMBOL], note2indexes)))

        extended_chorale = np.concatenate(
            (np.full(padding_dimensions, start_symbols),
             extended_chorale,
             np.full(padding_dimensions, end_symbols)),
            axis=0)
        chorale_length = len(extended_chorale)

        for time_index in range(chorale_length - timesteps):
            seq = chorale_to_onehot(
                extended_chorale[time_index:time_index + timesteps, :],
                num_pitches=num_pitches)
            X_onehot.append(seq)
    X_onehot = np.array(X_onehot)
    dataset = X_onehot, index2notes, note2indexes
    pickle.dump(dataset, file=open(pickled_onehot_dataset, 'wb'))


def find_nearest(features, hidden_repr_model, generator, num_elements=1000):
    """

    :param num_elements:
    :param hidden_repr_model:
    :param features:
    :param generator:
    :return: features produced by generator that achieve minimum distance
    """
    # max_dist = -10
    max_dist = 1000

    min_chorale = features
    hidden_repr = hidden_repr_model.predict(features, batch_size=1)
    intermediate_results = []
    for features_gen in tqdm(islice(generator, num_elements)):
        hidden_repr_gen = hidden_repr_model.predict(features_gen[0],
                                                    batch_size=1)

        # dist = spearmanr(hidden_repr[0], hidden_repr_gen[0])[0]
        dist = spearman_rho(hidden_repr[0], hidden_repr_gen[0])

        # if dist > max_dist:
        if dist < max_dist:
            max_dist = dist
            min_chorale = features_gen
            print(max_dist)

            if dist < 100:
                intermediate_results.append(min_chorale)

    intermediate_results.append(min_chorale)
    print(max_dist)
    return min_chorale, intermediate_results


def find_furthest(features, hidden_repr_model, generator):
    """

    :param features:
    :param model:
    :param generator:
    :return: (({'left_features': left_features,
                         'beats_left': beat_left,
                         'fermatas_left': fermata_left},
                        {'pitch_prediction': label})
    """
    min_dist = 1000
    min_chorale = features
    hidden_repr = hidden_repr_model.predict(features, batch_size=1)
    for features_gen in islice(generator, 10000):
        hidden_repr_gen = hidden_repr_model.predict(features_gen[0],
                                                    batch_size=1)
        dist = spearmanr(hidden_repr[0], hidden_repr_gen[0])[0]

        if dist < min_dist:
            min_dist = dist
            min_chorale = features_gen[0]['left_features'][0]

    print(min_dist)
    return min_chorale


def spearman_rho(v1, v2):
    assert len(v1.shape) == 1
    assert len(v1) == len(v2)
    l = len(v1)
    # apply function
    f = lambda x: x
    # f = lambda x: np.power(x, 1/2)
    squash = np.vectorize(lambda rk: f(l) if rk > l else f(rk))

    # compute ranks
    r1 = squash(rankdata(-v1, method='average'))
    r2 = squash(rankdata(-v2, method='average'))

    # print number of zeros
    # TODO understand why it's null
    # print('Spearman:')
    # print(sum(v1 > 1), sum(v2 > 1))
    # print(sum(v1 > 0), sum(v2 > 0))
    # print(sum(v1 == 0), sum(v2 == 0))
    # print(sum(v1 < 0), sum(v2 < 0))

    return np.sqrt(np.sum(np.square(r1 - r2)))


def permutation_distance(input_1, input_2, distance_model):
    return spearman_rho(distance_model.predict(input_1, batch_size=1)[0],
                        distance_model.predict(input_2, batch_size=2)[0])





def gibbs(generation_models=None, hidden_repr_model=None,
          inputs_target_chorale=None, chorale_metas=None,
          sequence_length=50, num_iterations=1000, timesteps=16,
          temperature=1., batch_size_per_voice=16,
          pickled_dataset=BACH_DATASET):
    """
    samples from models in model_base_name
    """

    X, X_metadatas, voices_ids, index2notes, note2indexes, metadatas = pickle.load(
        open(pickled_dataset, 'rb'))
    num_pitches = list(map(len, index2notes))
    num_voices = len(voices_ids)
    # load models if not
    if generation_models is None:
        raise ValueError

    # initialization sequence
    if chorale_metas is not None:
        sequence_length = len(chorale_metas[0])

    seq = np.zeros(shape=(2 * timesteps + sequence_length, num_voices))
    for expert_index in range(num_voices):
        # Add start and end symbol + random init
        seq[:timesteps, expert_index] = [note2indexes[expert_index][
                                             START_SYMBOL]] * timesteps
        seq[timesteps:-timesteps, expert_index] = np.random.randint(
            num_pitches[expert_index],
            size=sequence_length)

        seq[-timesteps:, expert_index] = [note2indexes[expert_index][
                                              END_SYMBOL]] * timesteps

    if chorale_metas is not None:
        # chorale_metas is a list
        extended_chorale_metas = [np.concatenate((np.zeros((timesteps,)),
                                                  chorale_meta,
                                                  np.zeros((timesteps,))),
                                                 axis=0)
                                  for chorale_meta in chorale_metas]

    else:
        raise NotImplementedError

    # # set target
    # hidden_repr_target = hidden_repr_model.predict({'input_seq': inputs_target_chorale['left_features']}, batch_size=1)[0]

    min_temperature = temperature
    temperature = 2.0
    min_voice = 0
    std_dist = 0.5
    # Main loop
    for iteration in tqdm(range(num_iterations)):

        temperature = max(min_temperature, temperature * 0.995)  # Recuit
        std_dist = min(2., std_dist * 1.010)
        print(std_dist)
        print(temperature)

        time_indexes = {}
        probas = {}

        # recompute target
        left_features_target, _, _, _ = all_features(seq,
                                                     voice_index=0,
                                                     time_index=16 * 9,
                                                     timesteps=32,
                                                     num_pitches=num_pitches,
                                                     num_voices=num_voices)
        hidden_repr_target = \
            hidden_repr_model.predict(
                {'input_seq': np.array([left_features_target])}, batch_size=1)[
                0]

        for voice_index in range(min_voice, num_voices):
            batch_input_features = []

            time_indexes[voice_index] = []

            for batch_index in range(batch_size_per_voice):
                time_index = np.random.randint(timesteps,
                                               sequence_length + timesteps)
                time_indexes[voice_index].append(time_index)

                (left_feature,
                 central_feature,
                 right_feature,
                 label) = all_features(seq, voice_index, time_index, timesteps,
                                       num_pitches, num_voices)

                left_metas, central_metas, right_metas = all_metadatas(
                    chorale_metadatas=extended_chorale_metas,
                    metadatas=metadatas,
                    time_index=time_index, timesteps=timesteps)

                input_features = {'left_features': left_feature[:, :],
                                  'central_features': central_feature[:],
                                  'right_features': right_feature[:, :],
                                  'left_metas': left_metas,
                                  'central_metas': central_metas,
                                  'right_metas': right_metas}

                # list of dicts: predict need dict of numpy arrays
                batch_input_features.append(input_features)

            # convert input_features
            batch_input_features = {key: np.array(
                [input_features[key] for input_features in
                 batch_input_features])
                                    for key in batch_input_features[0].keys()
                                    }
            # make all estimations
            probas[voice_index] = generation_models[voice_index].predict(
                batch_input_features,
                batch_size=batch_size_per_voice)

            # update
            for batch_index in range(batch_size_per_voice):
                probas_pitch = probas[voice_index][batch_index]
                dists = np.zeros_like(probas_pitch)

                # tweak probas with distance model
                if time_indexes[voice_index][batch_index] in range(16 * 6 - 32,
                                                                   16 * 6):
                    # test all indexes
                    # create batch for parallel updates
                    batch_current = []
                    for pitch_index in range(num_pitches[voice_index]):
                        # compute current

                        # todo copy only the portion needed
                        seq_current = seq.copy()
                        seq_current[time_indexes[voice_index][
                                        batch_index], voice_index] = pitch_index

                        left_features_current, _, _, _ = all_features(
                            seq_current,
                            voice_index=0, time_index=16 * 6,
                            timesteps=32,
                            num_pitches=num_pitches,
                            num_voices=num_voices)

                        left_metas_current, central_metas_current, _ = all_metadatas(
                            chorale_metadatas=extended_chorale_metas,
                            metadatas=metadatas,
                            time_index=16 * 6, timesteps=32)

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

                    # predict all hidden_repr in parallel
                    hidden_reprs_current = hidden_repr_model.predict(
                        batch_current, batch_size=len(batch_current))
                    # predict all distances
                    dists = np.array(list(map(
                        lambda hidden_repr_current: spearman_rho(
                            hidden_repr_current, hidden_repr_target),
                        hidden_reprs_current)))
                    # todo add sigma
                    # print('Before')
                    # print(dists.argmin(), probas_pitch.argmax())
                    # print(np.amin(dists), np.amax(dists))
                    normalized_dists = (dists - np.mean(dists)) / np.std(
                        dists) * std_dist
                    exp_dist = np.exp(-normalized_dists)

                    exp_dist /= np.sum(exp_dist)
                    exp_dist -= 1e-10
                    # print(np.min(exp_dist), np.max(exp_dist))
                else:
                    exp_dist = np.ones_like(probas_pitch)
                    exp_dist /= np.sum(exp_dist) - 1e-7

                # todo two temperatures!
                # use temperature
                probas_pitch = np.log(probas_pitch) / temperature
                probas_pitch = np.exp(probas_pitch) / np.sum(
                    np.exp(probas_pitch)) - 1e-7

                # combine both probability distributions
                probas_pitch *= exp_dist
                # probas_pitch = np.power(probas_pitch, 0.5) * np.power(exp_dist / np.sum(exp_dist), 0.5)
                probas_pitch = probas_pitch / np.sum(probas_pitch) - 1e-7

                # todo to remove

                # pitch can include slur_symbol
                pitch = np.argmax(np.random.multinomial(1, probas_pitch))

                seq[time_indexes[voice_index][
                        batch_index], voice_index] = pitch

    return seq[timesteps:-timesteps, :]


def test_autoencoder(model_name, timesteps, pickled_dataset=BACH_DATASET):
    voice_index = 0

    num_epochs = 200
    samples_per_epoch = 1024 * 100
    batch_size = 64
    nb_val_samples = 1024

    X, X_metadatas, voice_ids, index2notes, note2indexes, metadatas = pickle.load(
        open(pickled_dataset, 'rb'))
    # sequences
    num_voices = 1
    num_pitches = list(map(len, index2notes))

    generator_train = (({'left_features': left_features,
                         'central_features': central_features,
                         'right_features': right_features,
                         'left_metas': left_metas,
                         'right_metas': right_metas,
                         'central_metas': central_metas,
                         },
                        {'pitch_prediction': labels})
                       for (
                           (left_features, central_features, right_features),
                           (left_metas, central_metas, right_metas),
                           labels)

                       in generator_from_raw_dataset(batch_size=batch_size,
                                                     timesteps=timesteps,
                                                     voice_index=voice_index,
                                                     phase='train',
                                                     pickled_dataset=pickled_dataset
                                                     ))

    generator_unitary = (({'left_features': left_features,
                           'central_features': central_features,
                           'right_features': right_features,
                           'left_metas': left_metas,
                           'right_metas': right_metas,
                           'central_metas': central_metas,
                           },
                          {'pitch_prediction': labels})
                         for (
                             (left_features, central_features, right_features),
                             (left_metas, central_metas, right_metas),
                             labels)

                         in generator_from_raw_dataset(batch_size=1,
                                                       timesteps=timesteps,
                                                       voice_index=voice_index,
                                                       phase='all',
                                                       pickled_dataset=pickled_dataset
                                                       ))

    inputs, outputs = next(generator_train)

    model = load_model(model_name)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    hidden_repr_model = Model(input=model.input,
                              output=model.layers[-1].output)
    hidden_repr_model.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])

    # create score target
    chorale_seq = chorale_onehot_to_indexed_chorale(
        onehot_chorale=inputs['left_features'][0],
        num_pitches=num_pitches,
        time_major=False)

    score = indexed_chorale_to_score(chorale_seq,
                                     pickled_dataset=pickled_dataset)
    score.show()
    nearest_chorale_inputs, intermediate_results = find_nearest(inputs,
                                                                hidden_repr_model,
                                                                generator_unitary,
                                                                num_elements=20000
                                                                )
    # concat all results
    nearest_chorale = np.concatenate(
        [np.array(nearest_chorale_inputs[0]['left_features'][0])
         for nearest_chorale_inputs in intermediate_results],
        axis=0)

    # create score nearest
    nearest_chorale_seq = chorale_onehot_to_indexed_chorale(
        onehot_chorale=nearest_chorale,
        num_pitches=num_pitches,
        time_major=False)

    score_nearest = indexed_chorale_to_score(nearest_chorale_seq,
                                             pickled_dataset=pickled_dataset)
    score_nearest.show()


def create_models(model_name=None, create_new=False, num_dense=200,
                  num_units_lstm=[200, 200],
                  pickled_dataset=BACH_DATASET, num_voices=4, metadatas=None,
                  timesteps=16):
    """
    Choose one model
    :param model_name:
    :return:
    """

    _, _, _, index2notes, _, _ = pickle.load(open(pickled_dataset, 'rb'))
    num_pitches = list(map(len, index2notes))
    for voice_index in range(num_voices):
        # We only need one example for features dimensions
        gen = generator_from_raw_dataset(batch_size=1, timesteps=timesteps,
                                         voice_index=voice_index,
                                         pickled_dataset=pickled_dataset)

        (
            (left_features,
             central_features,
             right_features),
            (left_metas, central_metas, right_metas),
            labels) = next(gen)

        if 'skip' in model_name:
            model = skip_connections(num_features_lr=left_features.shape[-1],
                                     num_features_meta=left_metas.shape[-1],
                                     num_pitches=num_pitches[voice_index],
                                     num_dense=num_dense,
                                     num_units_lstm=num_units_lstm,
                                     timesteps=timesteps)
        if 'norelu' in model_name:
            model = skip_noRelu(num_features_lr=left_features.shape[-1],
                                num_features_meta=left_metas.shape[-1],
                                num_pitches=num_pitches[voice_index],
                                num_dense=num_dense,
                                num_units_lstm=num_units_lstm,
                                timesteps=timesteps)
        else:
            raise ValueError

        model_path_name = 'models/' + model_name + '_' + str(voice_index)
        if not os.path.exists(model_path_name + '.json') or create_new:
            model.save(model_path_name, overwrite=create_new)


def main():
    # parse arguments
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
                        type=int, default=1280)
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

    dataset_path = None
    pickled_dataset = BACH_DATASET

    # metadatas = [TickMetadatas(SUBDIVISION), FermataMetadatas(), KeyMetadatas(window_size=1)]
    metadatas = [TickMetadatas(SUBDIVISION), FermataMetadatas()]

    timesteps = args.timesteps
    batch_size = args.batch_size_train
    samples_per_epoch = args.samples_per_epoch
    nb_val_samples = args.num_val_samples
    num_units_lstm = args.num_units_lstm
    model_name = args.name.lower() + ext
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
    num_epochs = args.train
    overwrite = args.overwrite

    # Create pickled dataset
    if not os.path.exists(pickled_dataset):
        initialization(dataset_path,
                       metadatas=metadatas,
                       voice_ids=[SOP_INDEX],
                       BACH_DATASET=BACH_DATASET)

    # load dataset
    X, X_metadatas, voice_ids, index2notes, note2indexes, metadatas = pickle.load(
        open(pickled_dataset, 'rb'))

    # dataset dependant variables
    NUM_VOICES = len(voice_ids)
    num_voices = NUM_VOICES
    num_pitches = list(map(len, index2notes))
    num_iterations = args.num_iterations // batch_size_per_voice // num_voices

    # Create, train load models
    if not os.path.exists('models/' + model_name + '_' + str(
                    NUM_VOICES - 1) + '.yaml'):
        create_models(model_name, create_new=overwrite,
                      num_units_lstm=num_units_lstm, num_dense=num_dense,
                      pickled_dataset=pickled_dataset, num_voices=num_voices,
                      metadatas=metadatas,
                      timesteps=timesteps)
    if train:
        models = train_models(model_name=model_name,
                              samples_per_epoch=samples_per_epoch,
                              num_epochs=num_epochs,
                              nb_val_samples=nb_val_samples,
                              timesteps=timesteps,
                              pickled_dataset=pickled_dataset,
                              num_voices=NUM_VOICES, metadatas=metadatas,
                              batch_size=batch_size)
    else:
        models = load_models(model_name, num_voices=NUM_VOICES)

    # todo to remove
    # model_name = 'skip_large'
    # timesteps = 32
    #
    # test_autoencoder(model_name='models/' + model_name + '_0',
    #                  timesteps=timesteps,
    #                  pickled_dataset=pickled_dataset)

    distance_model = load_model('models/seq2seq_masking')
    # distance_model.compile(optimizer='adam', loss='categorical_crossentropy',
    #                        metrics=['accuracy'])

    hidden_repr_model = Model(input=distance_model.input,
                              output=distance_model.layers[1].output)
    hidden_repr_model.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])

    # create target
    left_features, _, _, _ = all_features(np.transpose(X[21], axes=(1, 0)),
                                          voice_index=0, time_index=16 * 4,
                                          timesteps=32,
                                          num_pitches=num_pitches,
                                          num_voices=num_voices)
    left_metas, central_metas, _ = all_metadatas(X_metadatas[21],
                                                 time_index=16 * 4,
                                                 timesteps=32,
                                                 metadatas=metadatas)

    inputs_target_chorale = {
        'left_features': np.array([left_features]),
        'left_metas': np.array([left_metas]),
        'central_metas': np.array([central_metas])
    }

    # show target
    score = indexed_chorale_to_score(X[21][:, 16 * 4 - 32: 16 * 4],
                                     pickled_dataset=pickled_dataset
                                     )
    score.show()

    generated_chorale = gibbs(generation_models=models,
                              hidden_repr_model=hidden_repr_model,
                              inputs_target_chorale=inputs_target_chorale,
                              chorale_metas=X_metadatas[12][:150],
                              num_iterations=200,
                              pickled_dataset=pickled_dataset,
                              timesteps=timesteps)

    # convert
    score = indexed_chorale_to_score(
        np.transpose(generated_chorale, axes=(1, 0)),
        pickled_dataset=pickled_dataset
        )
    score.show()


if __name__ == '__main__':
    # main()
    # exit()
    timesteps = 32
    batch_size = 128
    num_epochs = 3
    # create_onehot_corpus(16, BACH_DATASET_NO_META, BACH_ONEHOT_DATASET)

    XX, index2notes, note2indexes = pickle.load(
        open(BACH_ONEHOT_DATASET, 'rb'))
    print(XX.shape)
    num_features = XX.shape[-1]

    model = load_model('models/distance/seq2seq_masking')

    res = np.zeros((32, 1))
    for index in range(200, 230):
        target = XX[index]
        predictions = model.predict(XX[200, None], batch_size=1)[0]

        target = np.array([[np.argmax(slice)] for slice in target])
        predictions = np.array([[np.argmax(slice)] for slice in predictions])

        res = np.concatenate([res, target, predictions])
        # convert
        score = indexed_chorale_to_score(
            np.transpose(predictions, axes=(1, 0)),
            pickled_dataset='datasets/bach_sop_no_meta.pickle'
            )
        score.show()
        # convert
        score = indexed_chorale_to_score(np.transpose(target, axes=(1, 0)),
                                         pickled_dataset='datasets/bach_sop_no_meta.pickle'
                                         )
        score.show()
