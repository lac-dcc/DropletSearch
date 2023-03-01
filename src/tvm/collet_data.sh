#!/bin/bash

ARCH=(
    "x86_gogh"
    "x86_hokusai"
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

echo "" > data.csv
echo "" > time.csv
for ((k = 0; k < ${#ARCH[@]}; k++)); do
    echo ${ARCH[k]} >> time.csv
    for ((i = 0; i < ${#MODEL[@]}; i++)); do
        echo ${MODEL[i]} >> time.csv
        for ((j = 0; j < ${#TUNER[@]}; j++)); do
            echo ${TUNER[j]} >> time.csv
            #python3 script/get_best_config.py ${MODEL[i]} ${TUNER[j]} $ARCH >> data.csv
            python3 script/get_time.py results/${MODEL[i]}/${ARCH[k]}/${TUNER[j]}"/summary.log" >> time.csv
        done
    done
done
