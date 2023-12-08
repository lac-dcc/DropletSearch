import logging
import sys
import numpy as np 
import tvm
from tvm import te, topi, testing
from tvm.topi.testing import conv2d_nchw_python
import tvm.testing
import os
from tvm import autotvm
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from utils.parse_launch_config import parse_launch_config
from utils.get_best_config import get, get1
from utils.conv_cuda import execute
from tvm.topi.utils import traverse_inline


# logging config (for printing tuning log to screen)
#logging.getLogger("autotvm").setLevel(logging.DEBUG)
#logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

op1, op2 = None, None

def get_log_filename(N, CI, H, W, CO, KH, KW, strides, padding, path):
    suf = ""
    if op1 is not None:
        suf = suf + "_" + op1
    if op2 is not None:
        suf = suf + "_" + op2
    return os.path.join(path, "conv2d_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}{9}.log".format(N, CI, H, W, CO, KH, KW, strides, padding, suf))

@autotvm.template("conv2d_nchw.cuda")
def tvm_conv2d_nchw_tune_op(data, kernel, strides, padding, dilation, out_dtype):
    C = topi.nn.conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)
    conv = C
    add = None
    if op1 == "relu":
        C = topi.nn.relu(C)
    if op1 == "add":
        add = te.placeholder(C.shape, name='input2')
        C = topi.add(C, add)
    
    if op2 == "relu":
        C = topi.nn.relu(C)
    if op2 == "add":
        add = te.placeholder(C.shape, name='input2')
        C = topi.add(C, add)

    cfg = autotvm.get_config()
    s = te.create_schedule([C.op])

    def _callback(op):
        if op.tag == "conv2d_nchw":
            topi.cuda.schedule_direct_cuda(cfg, s, conv)

    traverse_inline(s, C.op, _callback)

    if add is None:
        return s, [data, kernel, C]
    else:
        return s, [data, kernel, add, C]

def tune_conv2d_nchw(N, CI, H, W, CO, KH, KW, strides, padding, path, n_trial=1000):
    log_filename = get_log_filename(N, CI, H, W, CO, KH, KW, strides, padding, path)
    data = te.placeholder((N, CI, H, W), name="input0")
    kernel = te.placeholder((CO, CI, KH, KW), name="input1")
    #task = autotvm.task.create("conv2d_nchw.cuda", args=(data, kernel, strides, padding, 1, "float32"), target='cuda')

    #C = topi.nn.conv2d_nchw(data, kernel, strides, padding, 1, "float32")

    task = autotvm.task.create(
         "conv2d_nchw.cuda", args=(data, kernel, strides, padding, 1, "float32"), target="cuda"
    )
    #print(task.config_space)


    # Use local gpu, measure 10 times for every config to reduce variance
    # The timeout of compiling a program is 10 seconds, the timeout for running is 4 seconds
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=1, min_repeat_ms=100, timeout=200),
    )

    # Begin tuning, log records to file `conv2d.log`
    # During tuning we will also try many invalid configs, so you are expected to
    # see many error reports. As long as you can see non-zero GFLOPS, it is okay.
    tuner = autotvm.tuner.GridSearchTuner(task)
    # n_trial = min(n_trial, len(task.config_space))
    
    tuner.tune(
        n_trial=n_trial,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(log_filename)],
    )

    # inspect the best config
    try:
        dispatch_context = autotvm.apply_history_best(log_filename)
        best_config = dispatch_context.query(task.target, task.workload)
    except:
        print("error: not valid shedule")
        print("best runtime: ", 0)
        print("compilation time: ", 0)
        exit(0)
    #print(log_filename)
    #print("\nBest config:")
    #print(best_config)

    #with dispatch_context:
    #    with tvm.target.Target('cuda'):
    #        s, arg_bufs = task.instantiate(best_config)
    #        print(arg_bufs)
    #        tir = str(tvm.lower(s, arg_bufs, simple_mode=True))
    #        func = tvm.build(s, arg_bufs, 'cuda', name='conv')
    #        
    #        # dev = tvm.cuda()
    #        # a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    #        # b_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    #        # c_np = np.random.uniform(size=tuple([int(c) for c in C.shape])).astype(np.float32)
    #
    #        # a_tvm = tvm.nd.array(a_np, device=dev)
    #        # b_tvm = tvm.nd.array(b_np, device=dev)
    #        # c_tvm = tvm.nd.array(c_np, device=dev)
    #        
    #        # func(a_tvm, b_tvm, c_tvm)
    #        # evaluator = func.time_evaluator(func.entry_name, dev, number=1000)
    #        # print("Time cost of this operator: %.10f" % evaluator(a_tvm, b_tvm, c_tvm).mean)
    #
    #        source_code = func.imported_modules[0].get_source()
    #        # kernel_filename = log_filename[:-4] + ".cc"
    #        kernel_filename = log_filename[:-4]
    #        # if op1 is not None:
    #        #     kernel_filename = kernel_filename + "_" + op1
    #        # if op2 is not None:
    #        #     kernel_filename = kernel_filename + "_" + op2
    #        kernel_filename = kernel_filename + ".cc"
    #
    #        grid, block = parse_launch_config(tir)
    #        launch_config_as_comment = "//"+"_".join(map(lambda x: str(x), grid + block)) + "\n"
    #        param = "//"+"_".join([str(N), str(CI), str(H), str(W), str(CO), str(KH), str(strides), str(padding)]) + "\n"
    #        for_nnfusion = "//dim3 grid(" + ", ".join(map(lambda x: str(x), grid)) + ");\n" + "//dim3 block(" + ", ".join(map(lambda x: str(x), block)) + ");\n"
    #        with open(kernel_filename, "w") as f:
    #            f.write(launch_config_as_comment + param + for_nnfusion + source_code)
    #        
    print("best runtime: ", get(log_filename)[0] * 1000)
    get1(log_filename)

def main():
    N, CI, H, W, CO, KH, KW, strides, dilation = [int(s) for s in sys.argv[1:10]]
    padding = sys.argv[10]
    global op1, op2
    path = ""
    
    for arg in sys.argv[11:]:
        if arg == "add":
            if op1 is None:
                op1 = arg
            else:
                op2 = arg
        elif arg == "relu":
            if op1 is None:
                op1 = arg
            else:
                op2 = arg
        else:
            path = arg
    # print(get_log_filename(N, CI, H, W, CO, KH, KW, strides, padding, path))
    print(N, CI, H, W, CO, KH, KW, strides, padding, path)
    tune_conv2d_nchw(N, CI, H, W, CO, KH, KW, strides, padding, path, n_trial=1)

main()