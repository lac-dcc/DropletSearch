import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.nn.utils import get_pad_tuple
import logging
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from utils.parse_launch_config import parse_launch_config
from utils.get_best_config import get, get1
from utils.pooling_cuda import execute

# logging config (for printing tuning log to screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))


def get_log_filename(pool_type, path, *shape):
    return os.path.join(path, "ansor_{0}_pooling_{1}.log".format(pool_type, "_".join([str(s) for s in shape])))

@auto_scheduler.register_workload
def pool_layer(pool_type, N, CI, H, W, KH, KW, strides, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    pool = topi.nn.pool2d(data, (KH, KW), (strides, strides), (1, 1), get_pad_tuple(padding, (KH, KW)), pool_type=pool_type)
    return [data, pool]

def tune_pool(pool_type, N, CI, H, W, KH, KW, strides, padding, path, n_trial=1000):
    log_filename = get_log_filename(pool_type, path, N, CI, H, W, KH, KW, strides, padding)
    target = tvm.target.Target("cuda")
    
    task = auto_scheduler.SearchTask(
        func=pool_layer, args=(pool_type, N, CI, H, W, KH, KW, strides, padding), target=target
    )

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=n_trial,  # change this to 1000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_filename)],
        verbose=2,
    )

    # Run auto-tuning (search)
    if not path:
        task.tune(tune_option)
    # Apply the best schedule
    sch, args = task.apply_best(log_filename)
    tir = str(tvm.lower(sch, args, simple_mode=True))
    source_code = task.print_best(log_filename, print_mode="cuda")
    
    kernel_filename = log_filename[:-4] + ".cc"
    grid, block = parse_launch_config(tir)
    launch_config_as_comment = "//"+"_".join(map(lambda x: str(x), grid + block)) + "\n"
    param = "//"+"_".join([pool_type, str(N), str(CI), str(H), str(W), str(KH), str(strides), str(padding)]) + "\n"
    for_nnfusion = "//dim3 grid(" + ", ".join(map(lambda x: str(x), grid)) + ");\n" + "//dim3 block(" + ", ".join(map(lambda x: str(x), block)) + ");\n"

    with open(kernel_filename, "w") as f:
        f.write(launch_config_as_comment + param + for_nnfusion + source_code)
    
    print("best runtime: ", get(log_filename)[0] * 1000)
    print("compilation time: ", get1(log_filename))

def main():
    N, CI, H, W, K, strides = tuple([int(s) for s in sys.argv[2:8]])
    padding = sys.argv[8]
    pool_type = sys.argv[1]
    path = sys.argv[9] if len(sys.argv) == 10 else ""
    # log_filename = get_log_filename(pool_type, path, N, CI, H, W, K, K, strides, padding)
    # filename = log_filename[:-4]

    # from tvm.topi.nn.utils import get_pad_tuple
    # padding = get_pad_tuple(padding, (K, K))[0]
    
    # data = te.placeholder((N, CI, H, W), name="data")
    # pool = topi.nn.pool2d(data, (K, K), (strides, strides), (1, 1), get_pad_tuple(padding, (K, K)), pool_type=pool_type)

    # res = "python3 ${{PARSE}} --op_type {6} --source_file {0}.cc --input0_shape {1} --output0_shape {2} --stride {3} --padding {4} --window_shape {5} --json_file={0}.json".format(
    #     filename, 
    #     " ".join(map(lambda x: str(x), [x for x in data.shape])), 
    #     " ".join(map(lambda x: str(x), [x for x in pool.shape])),
    #     " ".join(map(lambda x: str(x), [strides, strides])),
    #     " ".join(map(lambda x: str(x), [padding, padding])),
    #     " ".join(map(lambda x: str(x), [K, K])),
    #     "MaxPool" if pool_type == "max" else "AvgPool"
    #     )
    # res = "python3 ${{INJ}} {0}.json".format(filename)
    # print(res)
    print(pool_type, N, CI, H, W, K, strides, padding, path)
    tune_pool(pool_type, N, CI, H, W, K, K, strides, padding, path)

main()