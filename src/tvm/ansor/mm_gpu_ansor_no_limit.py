import numpy as np
import time
import tvm
from tvm import te, auto_scheduler

@auto_scheduler.register_workload
def matmul(N, L, M, dtype="float32"):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

    return [A, B, C]

if __name__ == "__main__":
    ## Create the search task
    target = tvm.target.Target("cuda")
    dev = tvm.cuda()
    N, L, M = 1000, 800, 700

    np.random.seed(0)
    a_np = np.random.uniform(size=(N, L)).astype(np.float32)
    b_np = np.random.uniform(size=(L, M)).astype(np.float32)
    c_np = a_np.dot(b_np)
    
    task = tvm.auto_scheduler.SearchTask(func=matmul, args=(N, L, M, "float32"), target=target)

    # Inspect the computational graph
    #print("Computational DAG:")
    #print(task.compute_dag)

    ## Set Parameters for Auto-Scheduler

    log_file = "gpu_matmul.json"
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=2000,  # change this to 20000 to achieve the best performance
        runner=auto_scheduler.LocalRunner(number=2, repeat=3),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=0
    )

    ## Run the search

    start = time.time()
    # Run auto-tuning (search)
    task.tune(tune_option)
    end = time.time()

    # Apply the best schedule
    sch, args = task.apply_best(log_file)

    ## Check correctness and evaluate performance
    with auto_scheduler.ApplyHistoryBest(log_file):
        func = tvm.build(sch, args, target)
        a_tvm = tvm.nd.array(a_np, device=dev)
        b_tvm = tvm.nd.array(b_np, device=dev)
        c_tvm = tvm.nd.array(c_np, device=dev)
        func(a_tvm, b_tvm, c_tvm)

    # Check results
    np.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-3)

    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, dev, number=10, repeat=3)
    eval = evaluator(a_tvm, b_tvm, c_tvm)
    print(", %f, %f, %f" % (eval.mean, eval.std, end-start))

    #print("Equivalent python schedule:")
    #print(task.print_best(log_file))
