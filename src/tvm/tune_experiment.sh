#!/bin/bash

ARCH="x86"
NICK="nitro5"

NAME=$ARCH"_"$NICK # please change this name for your machine

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

for ((i = 0; i < ${#MODEL[@]}; i++)); do
    mkdir -p results/${MODEL[i]}
    mkdir -p results/${MODEL[i]}/$NAME
    echo ${MODEL[i]}
    for ((j = 0; j < ${#TUNER[@]}; j++)); do
        mkdir -p results/${MODEL[i]}/$NAME/${TUNER[j]} 
        python3 script/tune_relay.py ${MODEL[i]} ${TUNER[j]} $NAME 0 > results/${MODEL[i]}/$NAME/${TUNER[j]}"/time_partial_summary.log"
    done
done
