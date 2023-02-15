#import d2ltvm
import numpy as np
import tvm
from tvm import te

# Save to the d2ltvm package
def matmul(n, m, l):
    """Return the computing expression of matrix multiplication
    A : n x l matrix
    B : l x m matrix
    C : n x m matrix with C = A B
    """
    k = te.reduce_axis((0, l), name='k')
    A = te.placeholder((n, l), name='A')
    B = te.placeholder((l, m), name='B')
    C = te.compute((n, m),
                    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
                    name='C')
    return A, B, C

row_A = 1000
col_A_B = 800
row_B = 700
A, B, C = matmul(row_A, row_B, col_A_B)
s = te.create_schedule(C.op)
print(tvm.lower(s, [A, B], simple_mode=True))
mod = tvm.build(s, [A, B, C])

