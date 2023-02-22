#!/bin/bash

ARCH="nitro5"

MODEL=(
    #"resnet-18"
    "vgg-16"
)

TUNER=(
    #"droplet" 
    #"gridsearch" 
    #"random" 
    "ga" 
    "xgb"
)

mkdir -p results

for ((i = 0; i < ${#MODEL[@]}; i++)); do
    mkdir -p results/${MODEL[i]}
    mkdir -p results/${MODEL[i]}/$ARCH
    for ((j = 0; j < ${#TUNER[@]}; j++)); do
        mkdir -p results/${MODEL[i]}/$ARCH/${TUNER[j]} 
        python3 script/tune_relay_x86.py ${MODEL[i]} ${TUNER[j]} $ARCH > results/${MODEL[i]}/$ARCH/${TUNER[j]}"/summary.log"
    done
done