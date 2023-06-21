import logging
import sys

import numpy as np
import tvm
import time
import os

from tvm import te, testing

num_threads = os.cpu_count()
os.environ["TVM_NUM_THREADS"] = str(num_threads)

def matmul(N, L, M, dtype="float32"):
    
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)
    
    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    
    s = te.create_schedule(C.op)

    return s, [A, B, C]

if __name__ == "__main__":

    N, L, M = 1500, 1100, 900

    dev = tvm.cpu()
    target = "llvm"

    np.random.seed(0)
    a_np = np.random.uniform(size=(N, L)).astype(np.float32)
    b_np = np.random.uniform(size=(L, M)).astype(np.float32)
    c_np = a_np.dot(b_np)

    # apply history best from log file
    with tvm.transform.PassContext(opt_level=0):
        with tvm.target.Target('llvm'):
            s, arg_bufs = matmul(N, L, M, "float32")
            func = tvm.build(s, arg_bufs)

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

    print("%.4f, %.4f" % (eval.mean, eval.std))
