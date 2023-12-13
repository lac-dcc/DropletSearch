rm -rf log
mkdir -p log
mkdir -p results
echo "matmul"
python3 src/matmul_tuning.py 16 1024 1024 log &> results/results_matmul.txt
grep "best runtime:" results/results_matmul.txt
grep "compilation time:" results/results_matmul.txt
echo "conv2d"
python3 src/conv2d_tuning.py 128 128 28 28 128 3 3 1 1 SAME log &> results/results_conv2d.txt
grep "best runtime:" results/results_conv2d.txt
grep "compilation time:" results/results_conv2d.txt
echo "depthwise"
python3 src/depthwise_tuning.py 128 84 83 83 5 5 2 1 SAME log &> results/results_depthwise.txt
grep "best runtime:" results/results_depthwise.txt
grep "compilation time:" results/results_depthwise.txt
echo "pooling"
python3 src/pooling_tuning.py avg 128 168 83 83 1 2 VALID log &> results/results_pool.txt
grep "best runtime:" results/results_pool.txt
grep "compilation time:" results/results_pool.txt
echo "reduce"
python3 src/reduction_tuning.py 128 512 1024 2 log &> results/results_reduce.txt
grep "best runtime:" results/results_reduce.txt
grep "compilation time:" results/results_reduce.txt
echo "relu"
python3 src/relu_tuning.py relu 4096 4096 &> results/results_relu.txt
grep "best runtime:" results/results_relu.txt
grep "compilation time:" results/results_relu.txt