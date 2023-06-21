import torch
import time
import numpy as np
import argparse


def conv1d_cuda(N, C, W, K, S, stride, padding, dilation, dtype):
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

    global RUN_NUMBER
    number, repeats = RUN_NUMBER

    for i in range(repeats):
        time_record = []
        for j in range(number):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            C_torch = torch.nn.functional.conv1d(
                A_torch, B_torch, bias=None, stride=stride, padding=padding, dilation=dilation
            )

            end.record()
            torch.cuda.synchronize()
            total = start.elapsed_time(end)
            time_record.append(total)
        if i == repeats - 1:
            mean_cost = np.mean(time_record)
    print(mean_cost)

test_shapes = [
    (16, 512, 892, 512, 3, 1, 2, 1),
    (16, 512, 892, 1024, 1, 1, 0, 1),
    (16, 1024, 892, 512, 1, 1, 0, 1),

    (16, 512, 892, 512, 3, 1, 4, 1),
    (16, 512, 892, 512, 3, 1, 8, 1),
    (16, 512, 892, 512, 3, 1, 16, 1),
    (16, 512, 892, 512, 3, 1, 32, 1),

]

def run_cuda():
    costs = []
    dtype = "FP16"
    for i, shape in enumerate(test_shapes):
        (N, C, W, K, S, stride, padding, dilation) = shape
        conv1d_cuda(N, C, W, K, S, stride, padding, dilation, dtype)
    print("cudnn: %s" % ("enabled" if torch.backends.cudnn.enabled else "disabled"))


example_text = """
    example:
        python conv1d.py --target cuda --enable_cudnn --number 5 --repeats 5 --begin 0 --num 10
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--enable_cudnn", action="store_true")
    parser.add_argument("--number", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument(
        "--begin", type=int, choices=list(range(len(test_shapes))), default=0
    )
    parser.add_argument(
        "--num",
        type=int,
        choices=list(range(1, len(test_shapes) + 1)),
        default=len(test_shapes),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["FP16", "FP32", "TF32", "FP64", "BF16", "INT8", "BOOL"],
        default="FP16",
    )
    parser.add_argument("--target", type=str, choices=["cuda", "llvm"], default="cuda")

    args = parser.parse_args()

    if args.enable_cudnn:
        assert torch.backends.cudnn.is_available()
        torch.backends.cudnn.enabled = True
    else:
        torch.backends.cudnn.enabled = False

    RUN_NUMBER = (args.number, args.repeats)

    args = parser.parse_args()
    run_cuda()
