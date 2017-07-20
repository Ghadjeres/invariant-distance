import argparse
import os

import pickle

from deepPermutations.data_preprocessing import \
    initialize_transposition_dataset
from deepPermutations.sequential_model import InvariantDistance
from deepPermutations.model_manager import ModelManager


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps',
                        help="model's range (default: %(default)s)",
                        type=int, default=32)
    parser.add_argument('-b', '--batch_size_train',
                        help=f'batch size used during training phase ('
                             f'default: %(default)s)',
                        type=int, default=128)
    parser.add_argument('-s', '--num_batch_samples',
                        help=f'number of batches per epoch (default: %('
                             f'default)s)',
                        type=int, default=100)
    parser.add_argument('--num_val_batch_samples',
                        help=f'number of validation batch samples '
                             f'(default: %(default)s)',
                        type=int, default=2560)
    parser.add_argument('-u', '--num_units_lstm', nargs='+',
                        help=f'number of lstm units (default: %(default)s)',
                        type=int, default=[200, 200])
    parser.add_argument('-d', '--num_dense',
                        help=f'size of non recurrent hidden layers '
                             f'(default: %(default)s)',
                        type=int, default=200)
    parser.add_argument('-n', '--name',
                        help='model name (default: %(default)s)',
                        choices=['relu', 'norelu'],
                        type=str, default='relu')
    parser.add_argument('-i', '--num_iterations',
                        help=f'number of gibbs iterations (default: %('
                             f'default)s)',
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
                        help=f'reharmonization of a melody from the corpus '
                             f'identified by its id',
                        type=int)
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    # Parse args
    args = get_arguments()
    if args.ext:
        ext = '_' + args.ext
    else:
        ext = ''
    timesteps = args.timesteps
    batch_size = args.batch_size_train
    batches_per_epoch = args.num_batch_samples
    nb_val_batch_samples = args.num_val_batch_samples

    num_units_lstm = args.num_units_lstm
    model_name = args.name.lower()
    sequence_length = args.length
    batch_size_per_voice = args.parallel
    num_units_lstm = args.num_units_lstm
    num_dense = args.num_dense

    train = args.train > 0
    num_epochs = args.train
    overwrite = args.overwrite

    # create dataset if doesn't exist
    dataset_name = 'transpose/bach_sop'
    rel_pickle_filepath = f'datasets/{dataset_name}.pickle'
    pickle_filepath = f'deepPermutations/{rel_pickle_filepath}'
    if not os.path.exists(pickle_filepath):
        from DeepBach.metadata import *

        metadatas = [TickMetadatas(SUBDIVISION), FermataMetadatas()]
        initialize_transposition_dataset(
            dataset_dir=None,
            metadatas=metadatas)

    # _, _, index2notes, _, _ = pickle.load(open(pickle_filepath, 'rb'))

    # INVARIANT DISTANCE
    distance_model_kwargs = dict(
        reg='l2',
        # reg=None,
        dropout_prob=0.3,
        lambda_reg=1e-5,
        num_layers=2,
        num_units_lstm=512,
    )

    invariant_distance = InvariantDistance(
        dataset_name=pickle_filepath,
        timesteps=timesteps,
        num_pitches=55,
        **distance_model_kwargs
    )

    # gen = invariant_distance.generator(batch_size=batch_size,
    #                                    phase='all',
    #                                    )
    # next(gen)

    model_manager = ModelManager(model=invariant_distance,
                                 lr=1e-4,
                                 lambda_reg=1.e-2
                                 )
    model_manager.load()
    if train:
        model_manager.train_model(batch_size=batch_size,
                                  num_epochs=num_epochs,
                                  batches_per_epoch=batches_per_epoch,
                                  plot=True,
                                  save_every=2,
                                  reg_norm=2
                                  )

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
    # invariant_distance_model.show_distance_matrix(chorale_index=241,
    #                                               time_index=32,
    #                                               show_plotly=True)
    # invariant_distance_model.show_mean_distance_matrix(chorale_index=0,
    #                                                    show_plotly=True)
    # invariant_distance_model.compute_stats(chorale_index=0, num_elements=1000)
    # invariant_distance_model.show_all_absolute_preds(effective_timestep=32)
    exit()
