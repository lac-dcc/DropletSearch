import logging
import sys

import numpy as np
import tvm

from tvm import autotvm, te, testing

@autotvm.template("template_matmul")
def matmul(N, L, M, search_space, dtype="float", order="ijk"):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    # get the config object
    cfg = autotvm.get_config()

    # define search space
    cfg.define_knob("tile_y", search_space)
    cfg.define_knob("tile_x", search_space)

    # schedule according to config
    yo, yi = s[C].split(y, cfg["tile_y"].val)
    xo, xi = s[C].split(x, cfg["tile_x"].val)

    if order == "ijk":
        s[C].reorder(xo, xi, yo, yi, k)
    elif order == "ikj":
        s[C].reorder(xo, xi, k, yo, yi)
    elif order == "jik":
        s[C].reorder(yo, yi, xo, xi, k)
    elif order == "jki":
        s[C].reorder(yo, yi, k, xo, xi)
    elif order == "kij":
        s[C].reorder(k, xo, xi, yo, yi)
    elif order == "kji":
        s[C].reorder(k, yo, yi, xo, xi)

    return s, [A, B, C]

if __name__ == "__main__":

    N, L, M = 1000, 800, 700
    search_space = [1] + [i for i in range(8,129,8)]

    order = ["ijk", "ikj", "jik", "jki", "kij", "kji"]
    dev = tvm.cpu()

    for ord in order:

        save_log = "example.log"  #"matmul_%s.log" % ord
        task = autotvm.task.create("template_matmul", args=(N, L, M, search_space, "float32", "ijk"), target="llvm")
        #print(task.config_space)

        n_trial = max(10, len(task.config_space))

        #logging.getLogger("autotvm").setLevel(logging.DEBUG)
        logging.getLogger("autotvm").setLevel(logging.ERROR)
        logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

        measure_option = autotvm.measure_option(builder="local", runner=autotvm.LocalRunner(number=2, repeat=5))

        #tuner = autotvm.tuner.RandomTuner(task)
        tuner = autotvm.tuner.DropletTuner(task)
        #tuner = autotvm.tuner.GridSearchTuner(task)

        tuner.tune(
            n_trial=n_trial,
            measure_option=measure_option,
            callbacks=[autotvm.callback.log_to_file(save_log)],
        )

        break

        # inspect the best config
        dispatch_context = autotvm.apply_history_best(save_log)
        best_config = dispatch_context.query(task.target, task.workload)
        print("Best config:", best_config, end="")

        # apply history best from log file
        with autotvm.apply_history_best(save_log):
            with tvm.target.Target("llvm"):
                s, arg_bufs = matmul(N, L, M, search_space, "float32")
                func = tvm.build(s, arg_bufs)

        # check correctness
        a_np = np.random.uniform(size=(N, L)).astype(np.float32)
        b_np = np.random.uniform(size=(L, M)).astype(np.float32)
        c_np = a_np.dot(b_np)

        a_tvm = tvm.nd.array(a_np)
        b_tvm = tvm.nd.array(b_np)
        c_tvm = tvm.nd.empty(c_np.shape)
        func(a_tvm, b_tvm, c_tvm)

        tvm.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-4)

        # Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
        # and the overhead of kernel launch. You can also use nvprof to validate the result.
        evaluator = func.time_evaluator(func.entry_name, dev, number=10, repeat=3)
        time = evaluator(a_tvm, b_tvm, c_tvm)
        print(", %f, %f, %s" % (time.mean, time.std, ord))
