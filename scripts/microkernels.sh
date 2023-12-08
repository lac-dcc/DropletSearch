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
    bash src/microkernels/${TOOL[j]}/run.sh
done