import tvm, time
import logging
import sys
from tvm import autotvm
from tvm import te, topi, testing
import json
import os

#sys.path.append("..")
#from utils.parse_launch_config import parse_launch_config
from utils.get_best_config import get
from utils.matmul_cuda import execute

#logging.getLogger('autotvm').setLevel(logging.DEBUG)
#logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

def get_log_filename(M, K, N, path):
    return os.path.join(path, "matmul_{0}_{1}_{2}.log".format(M, K, N))

def search_matmul_config(batch, in_dim, out_dim, path, tc=False, n_trial=10):
    data = te.placeholder((batch, in_dim), name='A', dtype="float32")
    weight = te.placeholder((in_dim, out_dim), name='B', dtype="float32")

    schedule = "dense_small_batch.gpu" if batch == 1 else "dense_large_batch.gpu"
    # schedule = "dense_small_batch.gpu"
    if tc:
        schedule = "dense_tensorcore.cuda"
    print(schedule)

    task = autotvm.task.create(schedule, args=(
        data, weight), target='cuda')
    print(task.config_space)
    
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(min_repeat_ms=100, timeout=200)
    )

    log_filename = get_log_filename(batch, in_dim, out_dim, path)

    tuner = autotvm.tuner.XGBTuner(task)
    # n_trial = min(n_trial, len(task.config_space))
    begin_start = time.time()
    if not path:
        tuner.tune(n_trial=n_trial, measure_option=measure_option,
                callbacks=[autotvm.callback.log_to_file(log_filename)])
    final_time = time.time() - begin_start

    # inspect the best config
    dispatch_context = autotvm.apply_history_best(log_filename)
    best_config = dispatch_context.query(task.target, task.workload)

    print(log_filename)
    print("\nBest config:")
    print(best_config)
            
    print("best runtime:", get(log_filename)[0] * 1000, " space time", final_time)

def main():
    batch, in_dim, out_dim = [int(s) for s in sys.argv[1:4]]
    path = sys.argv[4] if len(sys.argv) == 5 else ""
    print(batch, in_dim, out_dim, path)
    search_matmul_config(batch, in_dim, out_dim, path)

main()