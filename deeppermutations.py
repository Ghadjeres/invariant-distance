import argparse
import os

from deepPermutations.data_preprocessing import \
    initialize_transposition_dataset
from deepPermutations.model_manager import ModelManager
from deepPermutations.sequential_model import Distance, InvariantDistance


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps',
                        help="model's range (default: %(default)s)",
                        type=int, default=32)
    parser.add_argument('-b', '--batch_size_train',
                        help=f'batch size used during training phase ('
                             f'default: %(default)s)',
                        type=int, default=128)
    parser.add_argument('-B', '--batches_per_epoch',
                        help=f'number of batches per epoch (default: %('
                             f'default)s)',
                        type=int, default=100)
    parser.add_argument('-l', '--num_layers',
                        help=f'number of LSTM layers (default: %('
                             f'default)s)',
                        type=int, default=2)
    parser.add_argument('--num_val_batch_samples',
                        help=f'number of validation batch samples '
                             f'(default: %(default)s)',
                        type=int, default=2560)
    parser.add_argument('-u', '--num_lstm_units',
                        help=f'number of lstm units (default: %(default)s)',
                        type=int, default=256)
    parser.add_argument('-d', '--input_dropout',
                        help=f'dropout on input (default: %(default)s)',
                        type=float, default=0.001)
    parser.add_argument('-D', '--dropout_lstm',
                        help=f'dropout between LSTM layers (default: %('
                             f'default)s)',
                        type=float, default=0.001)
    parser.add_argument('-i', '--invariant', nargs='?',
                        help=f'if True, use transposition-invariant distance '
                             f'model',
                        default=False, const=True)
    parser.add_argument('-t', '--train', nargs='?',
                        help='train models for N epochs (default: 15)',
                        default=0, const=15, type=int)
    parser.add_argument('--overwrite',
                        help='overwrite previously computed models',
                        action='store_true')
    parser.add_argument('--dataset', nargs='?',
                        help='path to dataset folder',
                        type=str, default='')
    parser.add_argument('-r', '--ReLU', nargs='?',
                        help=f'add ReLU on hidden representation',
                        default=None, const='ReLU'
                        )
    parser.add_argument('-s', '--stats', nargs='?',
                        help=f'compute stats on N randomly drawn sequences',
                        default=0, const=10000, type=int
                        )
    parser.add_argument('-f', '--find-nearest', nargs='?',
                        help=f'find nearest neighbors',
                        default=0, const=10000, type=int
                        )
    parser.add_argument('-p', '--permutation_distance', nargs='?',
                        help=f'distance used in stats or nearest neighbors',
                        choices=['spearman', 'kendall', 'edit'],
                        default='spearman', type=str
                        )
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    # Parse args
    args = get_arguments()

    # training parameters
    batch_size = args.batch_size_train
    batches_per_epoch = args.batches_per_epoch
    nb_val_batch_samples = args.num_val_batch_samples
    train = args.train > 0
    num_epochs = args.train
    overwrite = args.overwrite

    # model parameters
    is_invariant = args.invariant
    timesteps = args.timesteps
    num_lstm_units = args.num_lstm_units
    non_linearity = args.ReLU
    input_dropout = args.input_dropout
    dropout_prob = args.dropout_lstm
    num_layers = args.num_layers

    # dataset parameters
    num_pitches = 55

    # visualizations
    compute_stats = args.stats > 0
    num_elements_stats = args.stats
    permutation_distance = args.permutation_distance

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

    # DISTANCE
    if is_invariant:
        distance = InvariantDistance(
            dataset_name=pickle_filepath,
            timesteps=timesteps,
            num_pitches=num_pitches,
            num_lstm_units=num_lstm_units,
            dropout_prob=dropout_prob,
            input_dropout=input_dropout,
            num_layers=num_layers,
            embedding_dim=16,
            non_linearity=non_linearity
        )
    else:
        distance = Distance(
            dataset_name=pickle_filepath,
            timesteps=timesteps,
            num_pitches=num_pitches,
            num_lstm_units=num_lstm_units,
            dropout_prob=dropout_prob,
            input_dropout=input_dropout,
            num_layers=num_layers,
            embedding_dim=16,
            non_linearity=non_linearity
        )

    model_manager = ModelManager(model=distance,
                                 lr=1e-3,
                                 lambda_reg=1.
                                 )
    model_manager.load()
    if train:
        model_manager.train_model(batch_size=batch_size,
                                  num_epochs=num_epochs,
                                  batches_per_epoch=batches_per_epoch,
                                  plot=True,
                                  save_every=2,
                                  reg_norm=None
                                  )

    if compute_stats:
        distance.compute_stats(
            num_elements=num_elements_stats,
            permutation_distance=permutation_distance,
            plot=True)

    exit()
    target_seq = distance.target_seq()

    distance.find_nearests_all(
        target_seq=target_seq,
        show_results=True,
        num_nearests=50,
        permutation_distance=permutation_distance
    )
    exit()

    distance.find_nearests(
        target_seq=None,
        show_results=True,
        num_elements=20000)
    # todo show pred
    # invariant_distance_model.test_transpose_out_of_bounds(
    #     effective_timestep=32)

    invariant_distance.show_mean_distance_matrix(chorale_index=241,
                                                 show_plot=True)
    # invariant_distance_model.show_mean_distance_matrix(chorale_index=0,
    #                                                    show_plotly=True)
    # invariant_distance_model.show_all_absolute_preds(effective_timestep=32)
    exit()
