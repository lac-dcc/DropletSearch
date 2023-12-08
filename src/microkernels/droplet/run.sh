mkdir -p log
mkdir -p results
python3 src/matmul_tuning.py 16 1024 1024 log #> results/results_matmul.txt
python3 src/conv2d_tuning.py 128 128 28 28 128 3 3 1 1 SAME log #> results/results_conv2d.txt
python3 src/depthwise_tuning.py 128 84 83 83 5 5 2 1 SAME log #> results/results_depthwise.txt
python3 src/pooling_tuning.py avg 128 168 83 83 1 2 VALID log #> results/results_pool.txt
python3 src/reduction_tuning.py 128 512 1024 2 log #> results/results_reduce.txt
python3 src/relu_tuning.py relu 4096 4096 #> results/results_relu.txt
