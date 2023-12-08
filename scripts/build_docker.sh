#!/bin/bash

help() 
{
    echo "Syntax: bash scripts/build_docker.sh [option]"
    echo "option:" 
    echo "  x86, arm, or cuda"
    exit 1
}

## MAIN
if [ $# -lt 1 ]; then
    help
else
    oper=$1
    if [ $oper == "help" ]; then
        help # call helper function
    elif [ $oper == "x86" ]; then
        docker build ./docker/x86/. -t droplet-x86
    elif [ $oper == "arm" ]; then
        docker build ./docker/arm32v7/. -t garkain-arm32v7
    elif [ $oper == "cuda" ]; then
        docker build ./docker/cuda/. -t droplet-cuda
    fi
fi