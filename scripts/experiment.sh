#!/bin/bash

set -e

CSV_INPUT=$1
python3 search_minimun_value.py $CSV_INPUT
python3 generate_matrix.py $CSV_INPUT
python3 print_graphic.py $CSV_INPUT
