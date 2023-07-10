mkdir -p OPS

python3 run.py -p tensorcore -c src/gemm.json
python3 run.py -p tensorcore -c src/c2d.json
