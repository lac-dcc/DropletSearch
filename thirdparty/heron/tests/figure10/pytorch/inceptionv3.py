import torch
import time
import numpy as np
import argparse


def conv2d_cuda(N, C, H, W, K, R, S, stride, padding, dilation, dtype):
    A_np = np.random.uniform(-10, 10, [N, C, H, W]).astype("float32")
    B_np = np.random.uniform(-10, 10, [K, C, R, S]).astype("float32")

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

            C_torch = torch.nn.functional.conv2d(
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
            [16, 149, 149, 32  ,32 , 3, 3, 1, 0, 1],
            [16, 147, 147, 32  ,64 , 3, 3, 1, 1, 1],
            [16, 73 , 73 , 64  ,80 , 1, 1, 1, 0, 1],
            [16, 73 , 73 , 80  ,192, 3, 3, 1, 0, 1],
            [16, 35 , 35 , 192 ,64 , 1, 1, 1, 0, 1],
            [16, 35 , 35 , 192 ,48 , 1, 1, 1, 0, 1],
            [16, 35 , 35 , 48  ,64 , 5, 5, 1, 2, 1],
            [16, 35 , 35 , 64  ,96 , 3, 3, 1, 1, 1],
            [16, 35 , 35 , 96  ,96 , 3, 3, 1, 1, 1],
                                   
            [16, 35 , 35 , 192 ,32 , 1, 1, 1, 0, 1],
            [16, 35 , 35 , 256 ,64 , 1, 1, 1, 0, 1],
            [16, 35 , 35 , 256 ,48 , 1, 1, 1, 0, 1],
            [16, 35 , 35 , 288 ,64 , 1, 1, 1, 0, 1],
            [16, 35 , 35 , 288 ,48 , 1, 1, 1, 0, 1],
            [16, 35 , 35 , 288 ,384, 3, 3, 2, 0, 1],
            [16, 35 , 35 , 96  ,96 , 3, 3, 2, 0, 1],
            [16, 17 , 17 , 768 ,192, 1, 1, 1, 0, 1],
            [16, 17 , 17 , 768 ,128, 1, 1, 1, 0, 1],
            [16, 17 , 17 , 128 ,128, 1, 7, 1, 0, 1],
                                   
            [16, 17 , 17 , 128 ,192, 7, 1, 1, 3, 1],
            [16, 17 , 17 , 128 ,128, 7, 1, 1, 3, 1],
            [16, 17 , 17 , 128 ,192, 1, 7, 1, 0, 1],
            [16, 17 , 17 , 768 ,160, 1, 1, 1, 0, 1],
            [16, 17 , 17 , 160 ,160, 1, 7, 1, 0, 1],
            [16, 17 , 17 , 160 ,192, 7, 1, 1, 3, 1],
            [16, 17 , 17 , 160 ,160, 7, 1, 1, 3, 1],
            [16, 17 , 17 , 160 ,192, 1, 7, 1, 0, 1],
            [16, 17 , 17 , 192 ,192, 1, 7, 1, 0, 1],
            [16, 17 , 17 , 192 ,192, 7, 1, 1, 3, 1],
                                   
            [16, 17 , 17 , 192 ,320, 3, 3, 2, 0, 1],
            [16, 17 , 17 , 192 ,192, 3, 3, 2, 0, 1],
            [16, 8  , 8  , 1280,320, 1, 1, 1, 0, 1],
            [16, 8  , 8  , 1280,384, 1, 1, 1, 0, 1],
            [16, 8  , 8  , 384 ,384, 1, 3, 1, 0, 1],
            [16, 8  , 8  , 384 ,384, 3, 1, 1, 1, 1],
            [16, 8  , 8  , 1280,448, 1, 1, 1, 0, 1],
            [16, 8  , 8  , 448 ,384, 3, 3, 1, 1, 1],
            [16, 8  , 8  , 1280,192, 1, 1, 1, 0, 1],
            [16, 8  , 8  , 2048,320, 1, 1, 1, 0, 1],

            [16, 8  , 8  , 2048,384, 1, 1, 1, 0, 1],
            [16, 8  , 8  , 2048,448, 1, 1, 1, 0, 1],
            [16, 8  , 8  , 2048,192, 1, 1, 1, 0, 1]

]



def run_cuda():
    global RUN_CONFIG
    batch, beg, num, dtype = RUN_CONFIG
    print("N, C, H, W, K, R, S, stride, padding, dilation, type, cost")
    
    costs = []
    for i, shape in enumerate(test_shapes[beg : beg + num]):
        (N, H, W, C, K, R, S, stride, padding, dilation) = shape
        conv2d_cuda(N, C, H, W, K, R, S, stride, padding, dilation, dtype)
    print("cudnn: %s" % ("enabled" if torch.backends.cudnn.enabled else "disabled"))

example_text = """
    example:
        python conv2d.py --target cuda --batch 256 --enable_cudnn --number 5 --repeats 5 --begin 0 --num 10 --dtype FP16
        python conv2d.py --target llvm --batch 1 --number 10 --repeats 10 --begin 0 --num 5 --dtype INT8
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch", type=int, default=16)
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
    RUN_CONFIG = (args.batch, args.begin, args.num, args.dtype)

    args = parser.parse_args()
    if args.target == "cuda":
        run_cuda()
