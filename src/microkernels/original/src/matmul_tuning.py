import tvm, time
import logging
import sys
from tvm import te, topi, testing
import json
import os
import inspect
import numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

def tvm_injective_mul_tune_op(*shape):
    data1 = te.placeholder(shape, name='input0')
    data2 = te.placeholder(shape, name='input1')
    out = topi.multiply(data1, data2)
    s = topi.cuda.injective.schedule_injective(out)
    return s, [data1, data2, out]

def tune_injective(t, shape, n_trial=1000):

    op = tvm_injective_mul_tune_op

    with tvm.target.Target("cuda"):
        s, arg_bufs = op(*shape)
        func = tvm.build(s, arg_bufs)
    
    dev = tvm.cuda()
    a_np = np.random.uniform(size=shape).astype(np.float32)
    b_np = np.random.uniform(size=shape).astype(np.float32)
    c_np = np.random.uniform(size=shape).astype(np.float32)

    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.array(c_np, device=dev)
    
    func(a_tvm, b_tvm, c_tvm)
    evaluator = func.time_evaluator(func.entry_name, dev, number=t)
    print("Time cost of this operator: %.10f" % evaluator(a_tvm, b_tvm, c_tvm).mean)

def main():
    t = 1000
    shape = (16,1024,1024)
    if len(sys.argv) > 2:
        t = sys.argv[1]
        shape = tuple([int(s) for s in sys.argv[2:]])
    print(t, shape)
    tune_injective(t, shape)

start_time = time.time()
main()
print("compilation time: %s" % (time.time() - start_time))