import logging
import sys

import numpy as np
import tvm
import time

from tvm import autotvm, te, testing

def get_best_time(log, ms=True):
    import json

    f = open(log, "r")
    best_avg = 9999.0
    best_std = 0.0
    for line in f.readlines():
        data = json.loads(line)
        r = np.mean(data["result"][0])
        if (best_avg > r):
            best_avg = r
            best_std = np.std(data["result"][0])
    f.close()

    if ms: # convet to ms
        best_avg *= 1000
        best_std *= 1000
    return best_avg, best_std

def mm(N, L, M, dtype="float32"):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

    return A, B, C

@autotvm.template("template_matmul")
def matmul(N, L, M, search_space, dtype="float32"):
    
    # Creating schedule
    A, B, C = mm(N, L, M, dtype=dtype)
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    # get the config object
    cfg = autotvm.get_config()

    # define search space
    cfg.define_knob("tile_x", search_space)
    cfg.define_knob("tile_y", search_space)

    #OL = s.cache_write(C, "local")

    # create cache stage
    #AA = s.cache_read(data, "shared", [OL])
    #AL = s.cache_read(AA, "local", [OL])
    
    # schedule according to config
    x0, x1 = s[C].split(x, cfg["tile_x"].val)
    y0, y1 = s[C].split(y, cfg["tile_y"].val)
    #k = s[C].split(loop=k, factors=[None, 8])

    # Bind GPU thread indices
    s[C].bind(x0, te.thread_axis("blockIdx.x"))
    s[C].bind(y0, te.thread_axis("blockIdx.y"))

    s[C].bind(x1, te.thread_axis("threadIdx.x"))
    s[C].bind(y1, te.thread_axis("threadIdx.y"))

    #s[C].compute_at(s[C], tx)

    #print(tvm.lower(s, [A, B, C]))
    
    cfg.define_knob("order", [0, 1, 2, 3, 4, 5])

    if cfg["order"].val == 0: # ijk
        s[C].reorder(x0, x1, y0, y1, k)
    elif cfg["order"].val == 1: # ikj
        s[C].reorder(x0, x1, k, y0, y1)
    elif cfg["order"].val == 2: # jik
        s[C].reorder(y0, y1, x0, x1, k)
    elif cfg["order"].val == 3: # jki
        s[C].reorder(y0, y1, k, x0, x1)
    elif cfg["order"].val == 4: # kij
        s[C].reorder(k, x0, x1, y0, y1)
    elif cfg["order"].val == 5: # kji
        s[C].reorder(k, y0, y1, x0, x1)

    return s, [A, B, C]

if __name__ == "__main__":

    N, L, M = 1000, 800, 700
    search_space = [1] + [i for i in range(8,129,8)]

    dev = tvm.cuda()
    target = "cuda"

    np.random.seed(0)
    a_np = np.random.uniform(size=(N, L)).astype(np.float32)
    b_np = np.random.uniform(size=(L, M)).astype(np.float32)
    c_np = a_np.dot(b_np)

    #tool = ["DropletTuner", "GridSearchTuner", "RandomTuner", "GATuner", "XGBTuner"]
    tool = ["DropletTuner"]

    for t in tool:

        save_log = "results/%s_gpu_mm.log" % (t)

        task = autotvm.task.create("template_matmul", args=(N, L, M, search_space, "float32"), target=target)

        #print(task.config_space)

        logging.getLogger("autotvm").setLevel(logging.ERROR)
        logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

        measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=autotvm.LocalRunner(number=2, repeat=5))

        n_trial = len(task.config_space)

        start = time.time()
        if t == "DropletTuner":
            tuner = autotvm.tuner.DropletTuner(task)
        elif t == "GridSearchTuner":
            tuner = autotvm.tuner.GridSearchTuner(task)
        elif t == "RandomTuner":
            tuner = autotvm.tuner.RandomTuner(task)
        elif t == "GATuner":
            tuner = autotvm.tuner.GATuner(task)
        elif t == "XGBTuner":
            tuner = autotvm.tuner.XGBTuner(task, loss_type="rank")

        tuner.tune(
            n_trial=n_trial,
            measure_option=measure_option,
            callbacks=[autotvm.callback.log_to_file(save_log)],
        )

        end = time.time()

        # inspect the best config
        dispatch_context = autotvm.apply_history_best(save_log)
        best_config = dispatch_context.query(task.target, task.workload)
        print("Best config:", best_config, end="")

        # apply history best from log file
        with autotvm.apply_history_best(save_log):
            with tvm.target.Target(target):
                s, arg_bufs = matmul(N, L, M, search_space, "float32")
                func = tvm.build(s, arg_bufs, target=target)

        # check correctness
        a_tvm = tvm.nd.array(a_np, device=dev)
        b_tvm = tvm.nd.array(b_np, device=dev)
        c_tvm = tvm.nd.empty(c_np.shape, device=dev)
        func(a_tvm, b_tvm, c_tvm)

        #tvm.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-4)

        # Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
        # and the overhead of kernel launch. You can also use nvprof to validate the result.
        evaluator = func.time_evaluator(func.entry_name, dev, number=10, repeat=3)
        eval = evaluator(a_tvm, b_tvm, c_tvm)

        best_avg, best_std = get_best_time(save_log)

        print(", %.4f, %.4f, %.4f, %.4f, %.2f" % (eval.mean, eval.std, best_avg, best_std, end-start))