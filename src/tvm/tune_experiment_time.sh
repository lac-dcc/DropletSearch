#!/bin/bash

set -e

ARCH="x86_nitro5" # please change this name for your machine

MODEL=(
    #"resnet-18"
    #"vgg-16"
    #"mobilenet"
    
)

TUNER=(
    "gridsearch" 
    #"random" 
    #"ga" 
    #"xgb"
    #"droplet" 
    #"ansor"
)

mkdir -p results

for ((i = 0; i < ${#MODEL[@]}; i++)); do
    mkdir -p results/${MODEL[i]}
    mkdir -p results/${MODEL[i]}/$ARCH
    for ((j = 0; j < ${#TUNER[@]}; j++)); do
        mkdir -p results/${MODEL[i]}/$ARCH/${TUNER[j]} 
        echo ${TUNER[j]} 
        if [ ${TUNER[j]} == "ansor" ]; then
            python3 script/ansor_network_x86.py ${MODEL[i]} $ARCH 1 
        else
            python3 script/tune_relay_x86.py ${MODEL[i]} ${TUNER[j]} $ARCH 1
        fi
    done
done
