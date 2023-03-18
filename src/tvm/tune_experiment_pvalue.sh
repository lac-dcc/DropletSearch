#!/bin/bash

ARCH="cuda"
NICK="gtx1650"
NAME=$ARCH"_"$NICK # please change this name for your machine

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
    "0.05"
    "0.10"
    "0.50"
    "1.00"
)

mkdir -p results

echo $NAME
for ((i = 0; i < ${#MODEL[@]}; i++)); do
    mkdir -p results/$NAME
    mkdir -p results/$NAME/${MODEL[i]}
    echo "* "${MODEL[i]}
    for ((j = 0; j < ${#TUNER[@]}; j++)); do
    	echo " -> "${TUNER[j]}
        mkdir -p results/$NAME/${MODEL[i]}/${TUNER[j]} 
        for ((k = 0; k < ${#PVALUE[@]}; k++)); do
            python3 script/tune_relay.py ${MODEL[i]} ${TUNER[j]} $NAME 0 $TRIALS $PVALUE > results/$NAME/${MODEL[i]}/${TUNER[j]}"/summary.log"
        done
    done
done
