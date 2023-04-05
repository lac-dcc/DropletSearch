from __future__ import annotations
import numpy as np
import tvm
from tvm.script import tir as T
import numpy as np 


@tvm.script.ir_module
class MatMulConv:
    @T.prim_func
    def main(A: T.Buffer[(64, 256), "float32"],
             B: T.Buffer[(3, 3), "float32"],
             C: T.Buffer[(61, 253), "float32"],
             X: T.Buffer[(64, 256), "float32"],
             Y: T.Buffer[(64, 256), "float32"],) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("X"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    X[vi, vj] = 0.0
                X[vi, vj] = X[vi, vj] + A[vi, vk] * Y[vk, vj]

        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                #kernel 3
                C[vi, vj] = X[vi, vj] * B[0, 0] + X[vi, vj+1] * B[0, 1] + X[vi, vj+2] * B[0, 2] \
                    + X[vi+1, vj] * B[1, 0] + X[vi+1, vj+1] * B[1, 1] + X[vi+1, vj+2] * B[1, 2] \
                        + X[vi+2, vj] * B[2, 0] + X[vi+2, vj+1] * B[2, 1] + X[vi+2, vj+2] * B[2, 2] 

def blocking(sch,
             tile_local_y1,
             tile_local_x1,
             tile_block_y1,
             tile_block_x1,
             tile_local_y2,
             tile_local_x2,
             tile_block_y2,
             tile_block_x2,
             tile_k):
    
    block_X = sch.get_block("X")
    block_C = sch.get_block("C")

    X_local = sch.cache_write(block_X, 0, "local")
    C_local = sch.cache_write(block_C, 0, "local")

    i, j, k = sch.get_loops(block=block_X)
    x, y, z = sch.get_loops(block=block_C)

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y1, tile_local_y1])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x1, tile_local_x1])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])
    sch.unroll(k1)
    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(X_local, j1)

    sch.decompose_reduction(block_X, k0)

    i0, i1, i2 = sch.split(loop=x, factors=[None, tile_block_y2, tile_local_y2])
    j0, j1, j2 = sch.split(loop=y, factors=[None, tile_block_x2, tile_local_x2])
    k0, k1 = sch.split(loop=z, factors=[None, tile_k])
    sch.unroll(k1)
    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)

    sch.decompose_reduction(block_C, k0)

    return sch


if __name__=="__main__":
    A_np = np.random.uniform(size=(64, 256)).astype("float32")
    Y_np = np.random.uniform(size=(64, 256)).astype("float32")

    dev = tvm.cpu(0)
    target = "arm"

    interval = [1]+list(range(8,129,8))
    for split_a in interval:
        for split_b in interval:
            sch = tvm.tir.Schedule(MatMulConv)
            sch = blocking(sch, split_a, split_a, split_a, split_a, split_b, split_b, split_b, split_b, 8)
            
            #pp.pprint(sch.mod.show())
            rt_mod = tvm.build(sch.mod, target=target)
            B_nd = tvm.nd.array(np.array([-2,-1,0,-1,1,1,0,1,2], dtype="float32").reshape((3, 3)), dev)
            
            #B_nd = tvm.nd.array(B_np, dev)
            C_nd = tvm.nd.array(np.zeros((61,253), dtype="float32"), dev)
            X_nd = tvm.nd.array(np.zeros((64,256), dtype="float32"), dev)
            Y_nd = tvm.nd.array(Y_np, dev)
            A_nd = tvm.nd.array(A_np, dev)
            
            evaluator = rt_mod.time_evaluator("main", dev, number=5, repeat=3)
            eval = evaluator(A_nd, B_nd, C_nd, X_nd, Y_nd)
            print("%d,%d,%.4f,%.4f" %(split_a, split_b, eval.mean, eval.std))
            