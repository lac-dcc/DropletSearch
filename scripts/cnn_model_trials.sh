#!/bin/bash

set -e

ARCH=$1
NICK="gtx1650" # please change this name for your machine
NAME=$ARCH"_"$NICK 

TRIALS=(
    100
    500
    1000
    2500
    5000
    10000
)

MODEL=(
    "resnet-18"
    "vgg-16"
    "mobilenet"
    "mxnet"
    "inception_v3"
    "bert"
)

TUNER=(
    "droplet" 
    "gridsearch" 
    "random" 
    "ga" 
    "xgb"
    "ansor"
)

echo $NAME
for ((i = 0; i < ${#MODEL[@]}; i++)); do
    echo "* "${MODEL[i]}
    DATA="results/"$NAME"/"$NAME"_"${MODEL[i]}".csv"
    echo "time,droplet,gridsearch,random,ga,xgb,ansor," > $DATA
    for ((k = 0; k < ${#TRIALS[@]}; k++)); do
        echo "SIZE: "${TRIALS[k]}
        echo "" > tmp.txt
        for ((j = 0; j < ${#TUNER[@]}; j++)); do
            echo " -> "${TUNER[j]}
            echo "-- "${TUNER[j]} > tmp.txt
            python3 src/tvm/script/tune_relay_trials.py ${MODEL[i]} ${TUNER[j]} $NAME ${TRIALS[k]} > tmp.txt
            cat tmp.txt >> $NAME"_results.csv"  
        done
        python3 script/split_time.py tmp.txt ${TRIALS[k]} #>> $DATA
    done
done

rm -rf tmp.txt