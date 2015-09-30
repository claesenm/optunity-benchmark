#!/bin/bash

STARTIDX=$1
STOPIDX=$2

for IDX in `seq $STARTIDX $STOPIDX`
do
    sh run-all.sh
    sh copy-to-repeated.sh $IDX
done
