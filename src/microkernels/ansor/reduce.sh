mkdir -p log

python3 src/reduction_tuning.py 128 512 1024 2 log
#python3 $1/reduction_tuning.py 65536 1024 1 $2
#python3 $1/reduction_tuning.py 128 4032 11 11 2 $2
#python3 $1/reduction_tuning.py 128 2048 7 7 2 $2