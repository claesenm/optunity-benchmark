#!/bin/bash

NUMBER=$1

for i in $(ls results/);
do
    echo "Copying results/$i to results-repeated/$i-$NUMBER."
    cp results/$i results-repeated/$i-$NUMBER
done
