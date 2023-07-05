from __future__ import annotations
import pprint
import numpy as np
import tvm
from tvm.script import tir as T

pp = pprint.PrettyPrinter(width=40, compact=True, indent=4)

@tvm.script.ir_module
class MatMul:
    @T.prim_func
    def main(A: T.Buffer[(1024, 1024), "float32"],
             B: T.Buffer[(1024, 1024), "float32"],
             C: T.Buffer[(1024, 1024), "float32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

def blocking(sch,
             tile_local_y1,
             tile_local_x1,
             tile_block_y1,
             tile_block_x1,
             tile_k):
    block_C = sch.get_block("C")
    C_local = sch.cache_write(block_C, 0, "local")
    i, j, k = sch.get_loops(block=block_C)

    if tile_local_y1 == 0:
        tile_local_y1 = 1
    if tile_local_x1 == 0:
        tile_local_x1 = 1
    if tile_block_y1 == 0:
        tile_block_y1 = 1
    if tile_block_x1 == 0:
        tile_block_x1 = 1

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y1, tile_local_y1])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x1, tile_local_x1])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])
    sch.unroll(k1)
    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)

    sch.decompose_reduction(block_C, k0)

    return sch

if __name__=="__main__":
    A_np = np.random.uniform(size=(1024, 1024)).astype("float32")
    B_np = np.random.uniform(size=(1024, 1024)).astype("float32")

    dev = tvm.cpu(0)
    r = []
    for split_a in range(0,130,8):
        for split_b in range(0,130,8):
            sch = tvm.tir.Schedule(MatMul)
            if split_a != 0 or split_b != 0:
                sch = blocking(sch, split_a, split_a, split_b, split_b, 8)
            
            #pp.pprint(sch.mod.show())
            rt_mod = tvm.build(sch.mod, target="llvm")
            
            #B_nd = tvm.nd.array(B_np, dev)
            C_nd = tvm.nd.array(np.zeros((1024,1024), dtype="float32"), dev)
            A_nd = tvm.nd.array(A_np, dev)
            B_nd = tvm.nd.array(B_np, dev)
            
            evaluator = rt_mod.time_evaluator("main", dev, number=1, repeat=10)
            time = evaluator(A_nd, B_nd, C_nd)
            
            print("%.2f, %d, %d, %f" % (C_nd.numpy().sum(), split_a, split_b, time.mean))
            