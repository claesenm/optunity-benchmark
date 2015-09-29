#!/bin/bash

declare -a NAMES=("digits-0" "digits-1" "digits-2" "digits-3" "digits-4" "digits-5" "digits-6" \
    "digits-7" "digits-8" "digits-9" "covtype-1" "covtype-2" "covtype-3" "covtype-4" \
    "covtype-5" "covtype-6" "covtype-7" "ionosphere" "diabetes")

for NAME in "${NAMES[@]}"
do
    echo "=================================="
    echo "======"
    echo "====== NOW RUNNING $NAME"
    echo "======"
    echo "=================================="

    sh run-experiment.sh $NAME
done
