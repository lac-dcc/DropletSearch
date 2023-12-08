#!/bin/bash

ARCH="x86"
NICK="AMD3700X_p" # please change this name for your machine
NAME=$ARCH"_"$NICK 

TRIALS=10000

MODEL=(
    "resnet-18"
    "vgg-16"
    "mobilenet"
    "mxnet"
    "inception_v3"
)

TUNER=(
    "droplet" 
)

PVALUE=(
    "0.01"
    "0.05"
    "0.10"
    "0.25"
    "1.00"
)

mkdir -p results

echo $NAME
for ((i = 0; i < ${#MODEL[@]}; i++)); do
    for ((k = 0; k < ${#PVALUE[@]}; k++)); do
        NEW_NAME=$NAME"_"${PVALUE[k]}
        mkdir -p results/$NEW_NAME
        mkdir -p results/$NEW_NAME/${MODEL[i]}
        echo "* "${MODEL[i]}
        for ((j = 0; j < ${#TUNER[@]}; j++)); do
            echo " -> "${TUNER[j]}
            mkdir -p results/$NEW_NAME/${MODEL[i]}/${TUNER[j]} 
            python3 src/tvm/script/tune_relay.py ${MODEL[i]} ${TUNER[j]} $NEW_NAME 0 $TRIALS ${PVALUE[k]} > results/$NEW_NAME/${MODEL[i]}/${TUNER[j]}"/summary.log"
        done
    done
done
