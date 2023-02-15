import numpy as np
import time
import tvm
import os
from tvm import te, auto_scheduler

@auto_scheduler.register_workload
def matmul(N, L, M, dtype="float", order="ijk"):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    if order == "ijk": # ijk
        s[C].reorder(x, y, k)
    elif order == "ikj": # ikj
        s[C].reorder(x, k, y)
    elif order == "jik": # jik
        s[C].reorder(y, x, k)
    elif order == "jki": # jki
        s[C].reorder(y, k, x)
    elif order == "kij": # kij
        s[C].reorder(k, x, y)
    elif order == "kji": # kji
        s[C].reorder(k, y, x)

    return s, [A, B, C]

if __name__ == "__main__":
    ## Create the search task
    target = tvm.target.Target("llvm")
    dev = tvm.cpu()
    N, L, M = 1000, 800, 700

    np.random.seed(0)
    a_np = np.random.uniform(size=(N, L)).astype(np.float32)
    b_np = np.random.uniform(size=(L, M)).astype(np.float32)
    c_np = a_np.dot(b_np)

    orders = ["ijk"]

    for ord in orders:
        
        task = tvm.auto_scheduler.SearchTask(func=matmul, args=(N, L, M, "float32", ord), target=target)

        ## Set Parameters for Auto-Scheduler

        log_file = "matmul.json"

        os.remove(log_file)

        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=100,  # change this to 20000 to achieve the best performance
            runner=auto_scheduler.LocalRunner(number=2, repeat=5, enable_cpu_cache_flush=True),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=0
        )

        ## Run the search

        start = time.time()
        # Run auto-tuning (search)
        with tvm.transform.PassContext(opt_level=3):
            task.tune(tune_option)
        
        end = time.time()

        # Apply the best schedule
        sch, args = task.apply_best(log_file)

        ## Inspecting the Optimized Schedule

        #print("Lowered TIR:")
        #print(tvm.lower(sch, args, simple_mode=True))

        ## Check correctness and evaluate performance
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(opt_level=3):
                func = tvm.build(sch, args, target)
                a_tvm = tvm.nd.array(a_np, device=dev)
                b_tvm = tvm.nd.array(b_np, device=dev)
                c_tvm = tvm.nd.array(c_np, device=dev)
                func(a_tvm, b_tvm, c_tvm)

        # Check results
        np.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-3)

        with tvm.transform.PassContext(opt_level=3):
            # Evaluate execution time.
            evaluator = func.time_evaluator(func.entry_name, dev, number=10, repeat=3)
            eval = evaluator(a_tvm, b_tvm, c_tvm)
            print(", %f, %f, %f" % (eval.mean, eval.std, end-start))

        #print("Equivalent python schedule:")
        #print(task.print_best(log_file))
