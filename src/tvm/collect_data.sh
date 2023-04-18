#!/bin/bash

ARCH=(
    #"x86_AMD3900X"
    #"x86_inteli7-3770"
    #"cuda_rtx3080"
    #"cuda_gtx1650"
    #"arm_cortex-a7"
    "arm_cortex-a72"
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

for ((k = 0; k < ${#ARCH[@]}; k++)); do
    DATA="results/"${ARCH[k]}"/time.csv"
    DATA_ITER="results/"${ARCH[k]}"/number_iter.csv"
    echo "" > $DATA
    echo "" > $DATA_ITER
    for ((i = 0; i < ${#MODEL[@]}; i++)); do
        echo ${MODEL[i]} >> $DATA
        for ((j = 0; j < ${#TUNER[@]}; j++)); do
            echo ${TUNER[j]} >> $DATA
            #python3 script/get_best_config.py ${MODEL[i]} ${TUNER[j]} $ARCH >> data.csv
            python3 script/get_time.py results/${ARCH[k]}/${MODEL[i]}/${TUNER[j]}"/summary.log" >> $DATA
            #python3 script/get_iter_times_partial.py results/${ARCH[k]}/${MODEL[i]}/${TUNER[j]} >> $DATA_ITER
        done
        echo "" >> $DATA
        echo "" >> $DATA_ITER
    done
done


PVALUE=(
    "0.01"
    "0.05"
    "0.10"
    "0.25"
    "1.00"
)

DATA="results/p_value/time.csv"
echo "" > $DATA
for ((l = 0; l < ${#PVALUE[@]}; l++)) do
    echo "pvalue ${PVALUE[l]}" >> $DATA
    for ((k = 0; k < ${#ARCH[@]}; k++)); do
        for ((i = 0; i < ${#MODEL[@]}; i++)); do
            python3 script/get_time.py results/p_value/x86_hokusai_p_${PVALUE[l]}/${MODEL[i]}"/droplet/summary.log" >> $DATA
        done
    done
done

