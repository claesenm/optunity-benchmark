#!/bin/bash

NAME=$1

# ensure the workspace is clean before starting
make clean
rm -f /tmp/$NAME
rm -f /tmp/data.pkl
rm -f /tmp/results.pkl

# remove existing results of experiment
./clean-experiment.sh $NAME

# generate objective function and search space definitions
python $NAME.py

# run all solvers, aggregate and print results
make results/$NAME-all.pkl

# clean up
rm -f /tmp/data.pkl /tmp/results.pkl
