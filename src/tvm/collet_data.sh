#!/bin/bash

ARCH=(
    #"x86_AMD3900X"
    #"x86_gogh"
    #"x86_hokusai"
    #"cuda_rtx3080"
    "cuda_gtx1650"
    "arm_cortex-a7"
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
    DATA="results/"${ARCH[k]}"_time.csv"
    echo "" > $DATA
    for ((i = 0; i < ${#MODEL[@]}; i++)); do
        echo ${MODEL[i]} >> $DATA
        for ((j = 0; j < ${#TUNER[@]}; j++)); do
            echo ${TUNER[j]} >> $DATA
            #python3 script/get_best_config.py ${MODEL[i]} ${TUNER[j]} $ARCH >> data.csv
            python3 script/get_time.py results/${ARCH[k]}/${MODEL[i]}/${TUNER[j]}"/summary.log" >> $DATA
        done
    done
done
