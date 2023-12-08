#mkdir -p results
echo "matmul"
python3 src/matmul.py 16 1024 1024 1000 #> results/results_matmul.txt
echo "conv2d"
python3 src/conv2d.py 128 128 28 28 128 3 3 1 1 SAME #> results/results_conv2d.txt
echo "depthwise"
python3 src/depthwise.py 128 84 83 83 5 2 1 SAME 1000 #> results/results_depthwise.txt
echo "pooling"
python3 src/pool.py 128 168 83 83 1 2 VALID 1000 #> results/results_pool.txt
echo "reduce"
python3 src/reduce.py 0 1000 #> results/results_reduce.txt
echo "relu"
python3 src/relu.py 4096 #> results/results_relu.txt
