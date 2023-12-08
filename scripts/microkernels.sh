#!/bin/bash

TOOL=(
    "original"
    "droplet"
    "autotvm"
    "ansor"
    "tf"
)

output_results="results/"
for ((j = 0; j < ${#TOOL[@]}; j++)); do
    echo "-------------------------------------------------------------------------------"
    echo ${TOOL[j]}
    cd src/microkernels/${TOOL[j]}
    bash run.sh
    cd ../../../
    echo "-------------------------------------------------------------------------------"
done