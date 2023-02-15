import logging
import sys

import numpy as np
import tvm
import time
import os

from tvm import autotvm, te, testing

def mm(N, L, M, dtype="float32"):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)
    
    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

    return A, B, C

@autotvm.template("template_matmul")
def matmul(N, L, M, search_space, dev, dtype="float"):
    # function
    A, B, C = mm(N, L, M, dtype=dtype)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    # get the config object
    cfg = autotvm.get_config()

    # define search space
    cfg.define_knob("tile_x", search_space)
    cfg.define_knob("tile_y", search_space)   

    # schedule according to config
    x0, x1 = s[C].split(x, cfg["tile_x"].val)
    y0, y1 = s[C].split(y, cfg["tile_y"].val)
    
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

    dev = tvm.cpu()
    target = "llvm"

    np.random.seed(0)
    a_np = np.random.uniform(size=(N, L)).astype(np.float32)
    b_np = np.random.uniform(size=(L, M)).astype(np.float32)
    c_np = a_np.dot(b_np)

    #tool = ["DropletTuner", "GridSearchTuner", "RandomTuner", "GATuner", "XGBTuner"]
    tool = ["DropletTuner"]

    for t in tool:

        save_log = "results/%s_mm.log" % (t)
        if os.path.isfile(save_log):
            os.remove(save_log)

        with tvm.transform.PassContext(opt_level=3):
            task = autotvm.task.create("template_matmul", args=(N, L, M, search_space, dev, "float32"), target=target)

        print(task.config_space)

        #logging.getLogger("autotvm").setLevel(logging.disable)
        #logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

        measure_option = autotvm.measure_option(builder="local", runner=autotvm.LocalRunner(number=2, repeat=5, enable_cpu_cache_flush=True))

        start = time.time()

        with tvm.transform.PassContext(opt_level=3):
            
            if t == "DropletTuner":
                n_trial = len(task.config_space)
                tuner = autotvm.tuner.DropletTuner(task)
            elif t == "GridSearchTuner":
                n_trial = len(task.config_space)            # 100% of search space
                tuner = autotvm.tuner.GridSearchTuner(task)
            elif t == "RandomTuner":
                n_trial = int(len(task.config_space) * 0.3) # %30% of search space
                tuner = autotvm.tuner.RandomTuner(task)
            elif t == "GATuner":
                n_trial = int(len(task.config_space) * 0.3) # %30% of search space
                tuner = autotvm.tuner.GATuner(task)
            elif t == "XGBTuner":
                n_trial = int(len(task.config_space) * 0.3) # %30% of search space
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
        print("%s, Best config:" %(t), best_config, end="")

        # apply history best from log file
        with autotvm.apply_history_best(save_log):
            with tvm.transform.PassContext(opt_level=3):
                with tvm.target.Target('llvm'):
                    s, arg_bufs = matmul(N, L, M, search_space, "float32")
                    func = tvm.build(s, arg_bufs)

        # check correctness
        # check correctness
        a_tvm = tvm.nd.array(a_np, device=dev)
        b_tvm = tvm.nd.array(b_np, device=dev)
        c_tvm = tvm.nd.empty(c_np.shape, device=dev)
        func(a_tvm, b_tvm, c_tvm)

        tvm.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-4)

        # Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
        # and the overhead of kernel launch. You can also use nvprof to validate the result.
        evaluator = func.time_evaluator(func.entry_name, dev, number=10, repeat=3)
        eval = evaluator(a_tvm, b_tvm, c_tvm)
        print(", %f, %f, %f" % (eval.mean, eval.std, end-start))

        break
