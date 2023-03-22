#!/bin/bash

set -e

NAME="cuda_gtx1650" # please change this name for your machine

TRIALS=(
    100
    500
    1000
    2500
    5000
)

MODEL=(
    "resnet-18"
    "vgg-16"
    "mobilenet"
    "mxnet"
    "inception_v3"
)

TUNER=(
    "droplet" 
    "gridsearch" 
    "random" 
    "ga" 
    "xgb"
    "ansor"
)

mkdir -p results

echo $NAME
for ((k = 0; k < ${#TRIALS[@]}; k++)); do
    echo "SIZE: "${TRIALS[k]}
    DATA="results/"$NAME"_"${TRIALS[k]}".csv"
    echo "" > $DATA
    for ((i = 0; i < ${#MODEL[@]}; i++)); do
        echo "* "${MODEL[i]}
        for ((j = 0; j < ${#TUNER[@]}; j++)); do
            echo " -> "${TUNER[j]}
            echo ${TUNER[j]} >> $DATA
            python3 script/tune_relay_trials.py ${MODEL[i]} ${TUNER[j]} $NAME ${TRIALS[k]} >> $DATA
        done
    done
done
