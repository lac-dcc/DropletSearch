from __future__ import annotations
import numpy as np
import tvm, sys
from tvm import te
from tvm.script import tir as T
from PIL import Image
import numpy as np 

@tvm.script.ir_module
class Conv:
    @T.prim_func
    def main(A: T.Buffer[(1027, 1027), "float32"],
             B: T.Buffer[(3, 3), "float32"],
             C: T.Buffer[(1025, 1025), "float32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                #kernel 3
                C[vi, vj] = A[vi, vj] * B[0, 0] + A[vi, vj+1] * B[0, 1] + A[vi, vj+2] * B[0, 2] \
                    + A[vi+1, vj] * B[1, 0] + A[vi+1, vj+1] * B[1, 1] + A[vi+1, vj+2] * B[1, 2] \
                        + A[vi+2, vj] * B[2, 0] + A[vi+2, vj+1] * B[2, 1] + A[vi+2, vj+2] * B[2, 2]


def blocking_cpu(sch,
             tile_local_y,
             tile_local_x,
             tile_block_y,
             tile_block_x,
             tile_k):
    block_C = sch.get_block("C")
    C_local = sch.cache_write(block_C, 0, "local")

    i, j, k = sch.get_loops(block=block_C)

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])
    sch.unroll(k1)
    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)
    sch.decompose_reduction(block_C, k0)

    return sch

def blocking_gpu(sch,
             tile_local_y,
             tile_local_x,
             tile_block_y,
             tile_block_x,
             tile_k):
    block_C = sch.get_block("C")
    C_local = sch.cache_write(block_C, 0, "local")

    i, j, k = sch.get_loops(block=block_C)

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])
    sch.unroll(k1)
    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)

    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")

    sch.bind(i1, "threadIdx.y")
    sch.bind(j1, "threadIdx.x")
    sch.decompose_reduction(block_C, k0)

    return sch


if __name__=="__main__":
    
    arch = "x86"
    if len(sys.argv) > 1:
        arch = sys.argv[1]

    im = Image.open("./data/test3.jpg")
   
    A_np = np.array(im)
    kernelValues = np.array([-2,-1,0,-1,1,1,0,1,2]) #emboss
    kernel = kernelValues.reshape((3, 3))
    
    if "x86" in arch:
        target = "llvm"
    elif "cuda" in arch:
        target = "cuda"
    elif "arm" in arch:
        target = "llvm -device=arm_cpu"
    else:
        print("Arch not found! Available: x86, arm, or cuda")
        exit(0)
    
    dev = tvm.device(str(target), 0)
    
    if "cuda" in arch:
        interval = [1]+list(range(2,32,2))
    else:
        interval = [1]+list(range(8,129,8))

    for i in interval:
        for j in interval:
            sch = tvm.tir.Schedule(Conv)
            if arch == "cuda":
                sch = blocking_gpu(sch, i, i, j, j, 8)
            else:
                sch = blocking_cpu(sch, i, i, j, j, 8)
            
            rt_mod = tvm.build(sch.mod, target=target)
            A_np = np.random.uniform(size=(1027,1027)).astype("float32")
            B_nd = tvm.nd.array(np.array([-2,-1,0,-1,1,1,0,1,2], dtype="float32").reshape((3, 3)), dev)
            C_nd = tvm.nd.array(np.zeros((1025,1025), dtype="float32"), dev)

            A_nd = tvm.nd.array(A_np, dev)
            
            evaluator = rt_mod.time_evaluator("main", dev, number=10, repeat=3)
            eval = evaluator(A_nd, B_nd, C_nd)
            print("%d,%d,%.6f,%.6f" %(i, j, eval.mean, eval.std))