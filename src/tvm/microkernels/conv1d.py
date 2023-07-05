import torch
import time
import numpy as np
import argparse
import tvm
from tvm import autotvm, auto_scheduler, te, relay
from tvm.relay import testing

def conv1d(N, C, W, K, S, stride, padding, dilation, dtype="FP16"):
    A_np = np.random.uniform(-10, 10, [N, C, W]).astype("float32")
    B_np = np.random.uniform(-10, 10, [K, C, S]).astype("float32")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if dtype == "FP16":  # HMMA-16, torch.float16 or torch.half
        A_torch = torch.tensor(A_np).type(torch.float16).cuda()
        B_torch = torch.tensor(B_np).type(torch.float16).cuda()
    elif dtype == "BF16":  # HMMA-16, only on NVIDIA A100, torch.bfloat16
        A_torch = torch.tensor(A_np).type(torch.bfloat16).cuda()
        B_torch = torch.tensor(B_np).type(torch.bfloat16).cuda()
    elif dtype == "FP32":
        A_torch = torch.tensor(A_np).type(torch.float32).cuda()
        B_torch = torch.tensor(B_np).type(torch.float32).cuda()
    elif dtype == "TF32":  # HMMA-19, NVIDIA A100
        # Please upgrade torch to 1.7; only supported on A100
        # https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        A_torch = torch.tensor(A_np).type(torch.float32).cuda()
        B_torch = torch.tensor(B_np).type(torch.float32).cuda()
    elif dtype == "INT8":  # IMMA, but pytorch has no support for INT8 GEMM
        A_torch = torch.tensor(A_np).type(torch.int8).cuda()
        B_torch = torch.tensor(B_np).type(torch.int8).cuda()
    # Pytorch has no int4 type
    elif dtype == "BOOL":  # BMMA, but pytorch has no support for GEMM GEMM
        A_torch = torch.tensor(A_np).type(torch.bool).cuda()
        B_torch = torch.tensor(B_np).type(torch.bool).cuda()
    elif dtype == "FP64":  # DMMA(FP64), only supported on A100
        A_torch = torch.tensor(A_np).type(torch.float64).cuda()
        B_torch = torch.tensor(B_np).type(torch.float64).cuda()
    else:
        assert False, "wrong type: " + dtype

    model = torch.nn.functional.conv1d(
        A_torch, B_torch, bias=None, stride=stride, padding=padding, dilation=dilation
    )

    scripted_model = torch.jit.trace(model, [A_torch, B_torch]).eval()
    shape_list = [('A_torch', [N, C, W]), ('B_torch', [K, C, S])]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    BindPass = tvm.relay.transform.function_pass(lambda fn, new_mod, ctx:
                            tvm.relay.build_module.bind_params_by_name(fn, params), opt_level=1)
    mod = BindPass(mod)

    return mod, params

shape = (16, 512, 892, 512, 3, 1, 2, 1)
(N, C, W, K, S, stride, padding, dilation) = shape

mod, params = conv1d(N, C, W, K, S, stride, padding, dilation)

target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)

with tvm.transform.PassContext(opt_level=3):
    graph, lib, params = relay.build(mod, target=target, params=params)