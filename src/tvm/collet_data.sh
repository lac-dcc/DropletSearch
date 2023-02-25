#!/bin/bash

set -e

ARCH="nitro5"

MODEL=(
    #"resnet-18"
    "vgg-16"
)

TUNER=(
    "droplet" 
    "gridsearch" 
    "random" 
    "ga" 
    "xgb"
)

mkdir -p results

echo "" > data.csv
echo "" > time.csv
for ((i = 0; i < ${#MODEL[@]}; i++)); do
    mkdir -p results/${MODEL[i]}
    mkdir -p results/${MODEL[i]}/$ARCH
    for ((j = 0; j < ${#TUNER[@]}; j++)); do
        mkdir -p results/${MODEL[i]}/$ARCH/${TUNER[j]} 
        python3 script/get_best_config.py ${MODEL[i]} ${TUNER[j]} $ARCH >> data.csv
        python3 script/get_time.py results/${MODEL[i]}/$ARCH/${TUNER[j]}/summary.log >> time.csv
    done
done