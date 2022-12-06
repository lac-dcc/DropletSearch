#!/bin/bash

COMPILER=gcc
MACHINE="ryzen_x86"

mkdir -p results
mkdir -p bin

ORDER=("ijk" "ikj" "jik" "jki" "kij" "kji")

for ((i = 0; i < ${#ORDER[@]}; i++)); do
    RESULTS="results/"$MACHINE"_"${ORDER[i]}".csv"
    echo "sum,A,B,t0,t1,t2,t3,t4,t5,t6,t7,t8,t9" > $RESULTS
    $COMPILER -O3 search_space_${ORDER[i]}.c matrix_lib.c -o bin/tile_${ORDER[i]}.out
    ./bin/tile_${ORDER[i]}.out >> $RESULTS
done
