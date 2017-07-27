#!/usr/bin/env bash
export PYTHONPATH=~/Projets/Python/workspace/DeepBach:$PYTHONPATH
python deeppermutations.py -b 256 -i -d 0.3 -D 0.5 -t 500
python deeppermutations.py -b 256 -i -r -d 0.3 -D 0.5 -t 500