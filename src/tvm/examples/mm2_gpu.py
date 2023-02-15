from __future__ import annotations
import pprint
import numpy as np
import tvm
from tvm.script import tir as T

pp = pprint.PrettyPrinter(width=40, compact=True, indent=4)

@tvm.script.ir_module
class MatMul:
    @T.prim_func
    def main(A: T.Buffer[(1000, 800), "float32"],
             B: T.Buffer[(800, 700), "float32"],
             C: T.Buffer[(1000, 700), "float32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1000, 800, 700):
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

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y1, tile_local_y1])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x1, tile_local_x1])
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
    A_np = np.random.uniform(size=(1000, 800)).astype("float32")
    B_np = np.random.uniform(size=(800, 700)).astype("float32")

    dev = tvm.cuda(0)

    for split_a in [1]:
        for split_b in [1]:
            sch = tvm.tir.Schedule(MatMul)
            #sch = blocking(sch, split_a, split_a, split_a, split_a, split_b, split_b, split_b, split_b, 8)
            
            pp.pprint(sch.mod.show())
            rt_mod = tvm.build(sch.mod, target="cuda")
            
            #B_nd = tvm.nd.array(B_np, dev)
            C_nd = tvm.nd.array(np.zeros((1000,700), dtype="float32"), dev)
            A_nd = tvm.nd.array(A_np, dev)
            B_nd = tvm.nd.array(B_np, dev)
            
            
            #evaluator = rt_mod.time_evaluator("main", dev, number=10)
            #print("MM_Conv-Blocking: mm: %s  conv: %s %f GFLOPS" % (split_a, split_b, (num_flop / evaluator(A_nd, B_nd, C_nd, X_nd, Y_nd).mean / 1e9)))
            #print(C_nd)