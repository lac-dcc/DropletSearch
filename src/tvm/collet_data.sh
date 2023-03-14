#!/bin/bash

ARCH=(
    "x86_AMD3900X"
    #"cuda_rtx3080"
    #"cuda_gtx1650"
    #"arm_cortex-a7"
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
    DATA_ITER="results/"${ARCH[k]}"/number_iter.csv"
    echo "" > $DATA
    echo "" > $DATA_ITER
    for ((i = 0; i < ${#MODEL[@]}; i++)); do
        echo ${MODEL[i]} >> $DATA
        for ((j = 0; j < ${#TUNER[@]}; j++)); do
            echo ${TUNER[j]} >> $DATA
            #python3 script/get_best_config.py ${MODEL[i]} ${TUNER[j]} $ARCH >> data.csv
            python3 script/get_time.py results/${ARCH[k]}/${MODEL[i]}/${TUNER[j]}"/summary.log" >> $DATA
            python3 script/get_iter_times_partial.py results/${ARCH[k]}/${MODEL[i]}/${TUNER[j]} >> $DATA_ITER
        done
    done
done


