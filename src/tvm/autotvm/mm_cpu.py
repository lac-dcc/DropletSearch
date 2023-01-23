import logging
import sys

import numpy as np
import tvm

from tvm import autotvm, te, testing

@autotvm.template("template_matmul")
def matmul(N, L, M, search_space, dtype):
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
    #cfg.define_split("tile_y", y, num_outputs=2)
    #cfg.define_split("tile_x", x, num_outputs=2)

    # define search space
    cfg.define_knob("tile_y", search_space)
    cfg.define_knob("tile_x", search_space)

    # schedule according to config
    yo, yi = s[C].split(y, cfg["tile_y"].val)
    xo, xi = s[C].split(x, cfg["tile_x"].val)
    #yo, yi = cfg["tile_y"].apply(s, C, y)
    #xo, xi = cfg["tile_x"].apply(s, C, x)

    s[C].reorder(xo, xi, yo, yi, k)

    return s, [A, B, C]

N, L, M = 1000, 800, 700
search_space = [1] + [i for i in range(8,129,8)]

task = autotvm.task.create("template_matmul", args=(N, L, M, search_space, "float32"), target="llvm")
print(task.config_space)

logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

measure_option = autotvm.measure_option(builder="local", runner=autotvm.LocalRunner(number=5))

tuner = autotvm.tuner.RandomTuner(task)
tuner.tune(
    n_trial=10,
    measure_option=measure_option,
    callbacks=[autotvm.callback.log_to_file("matmul.log")],
)

# inspect the best config
dispatch_context = autotvm.apply_history_best("matmul.log")
best_config = dispatch_context.query(task.target, task.workload)
print("\nBest config:")
print(best_config)

# apply history best from log file
with autotvm.apply_history_best("matmul.log"):
    with tvm.target.Target("llvm"):
        s, arg_bufs = matmul(N, L, M, search_space, "float32")
        func = tvm.build(s, arg_bufs)

# check correctness
a_np = np.random.uniform(size=(N, L)).astype(np.float32)
b_np = np.random.uniform(size=(L, M)).astype(np.float32)
c_np = a_np.dot(b_np)

c_tvm = tvm.nd.empty(c_np.shape)
func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)

tvm.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-4)