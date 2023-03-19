#!/bin/bash

set -e

NAME="cuda_gtx1650" # please change this name for your machine

TRIALS=10000

MODEL=(
    "resnet-18"
    "vgg-16"
    "mobilenet"
    "mxnet"
    "inception_v3"
)

TUNER=(
    "gridsearch" 
    "random" 
    "ga" 
    "xgb"
    "droplet" 
    "ansor"
)

mkdir -p results

echo $NAME
for ((i = 0; i < ${#MODEL[@]}; i++)); do
    echo "* "${MODEL[i]}
    for ((j = 0; j < ${#TUNER[@]}; j++)); do
        echo " -> "${TUNER[j]} 
        python3 script/tune_relay.py ${MODEL[i]} ${TUNER[j]} $NAME 1 $TRIALS
    done
done
