#!/usr/bin/env bash

#usage: deeppermutations.py [-h] [--timesteps TIMESTEPS] [-b BATCH_SIZE_TRAIN]
#                           [-B BATCHES_PER_EPOCH] [-L NUM_LAYERS]
#                           [--num_val_batch_samples NUM_VAL_BATCH_SAMPLES]
#                           [-u NUM_LSTM_UNITS] [-d INPUT_DROPOUT]
#                           [-D DROPOUT_LSTM] [-i [INVARIANT]] [-t [TRAIN]]
#                           [--overwrite] [--dataset [DATASET]] [-r [RELU]]
#                           [-s [STATS]] [-f [FIND_NEAREST]]
#                           [-p [{spearman,kendall,edit}]] [-l [L_TRUNCATION]]
#                           [--norm [{l1,l2}]]

export PYTHONPATH=~/Projets/Python/workspace/DeepBach:$PYTHONPATH
#python deeppermutations.py -b 256 -i -d 0.3 -D 0.5 -t 500
#python deeppermutations.py -b 256 -i -r -d 0.3 -D 0.5 -t 500

python deeppermutations.py -b 256 -i -r --norm l2 -d 0.3 -D 0.5 -t 100
python deeppermutations.py -b 256 -i -r --norm l2 -d 0.3 -D 0.5 -s -p spearman
python deeppermutations.py -b 256 -i -r --norm l2 -d 0.3 -D 0.5 -s -p kendall

# plot stats
#python deeppermutations.py -b 256 -i -d 0.3 -D 0.5 -s -p spearman
#python deeppermutations.py -b 256 -i -d 0.3 -D 0.5 -s -p kendall
#python deeppermutations.py -b 256 -i -r -d 0.3 -D 0.5 -s -p spearman
#python deeppermutations.py -b 256 -i -r -d 0.3 -D 0.5 -s -p kendall