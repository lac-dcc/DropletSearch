import torch
import argparse
import numpy as np

def bmm(B, H, W, K, dtype):
  A_np = np.random.uniform(-10, 10, [B, H, K]).astype("float32")
  B_np = np.random.uniform(-10, 10, [B, K, W]).astype("float32")

  # What's supported by NVIDIA? Refer to https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html

  # What's supported by pytorch? I don't know
  # Please sudo nvprof them!

  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False

  if dtype == "FP16": # HMMA-16, torch.float16 or torch.half
    A_torch = torch.tensor(A_np).type(torch.float16).cuda()
    B_torch = torch.tensor(B_np).type(torch.float16).cuda()
  elif dtype == "BF16": # HMMA-16, only on NVIDIA A100, torch.bfloat16
    A_torch = torch.tensor(A_np).type(torch.bfloat16).cuda()
    B_torch = torch.tensor(B_np).type(torch.bfloat16).cuda()
  elif dtype == "FP32":
    A_torch = torch.tensor(A_np).type(torch.float32).cuda()
    B_torch = torch.tensor(B_np).type(torch.float32).cuda()
  elif dtype == "TF32": # HMMA-19, NVIDIA A100
    # Please upgrade torch to 1.7; only supported on A100
    # https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    A_torch = torch.tensor(A_np).type(torch.float32).cuda()
    B_torch = torch.tensor(B_np).type(torch.float32).cuda()
  elif dtype == "INT8": # IMMA, but pytorch has no support for INT8 GEMM
    A_torch = torch.tensor(A_np).type(torch.int8).cuda()
    B_torch = torch.tensor(B_np).type(torch.int8).cuda()
  # Pytorch has no int4 type
  elif dtype == "BOOL": # BMMA, but pytorch has no support for GEMM GEMM
    A_torch = torch.tensor(A_np).type(torch.bool).cuda()
    B_torch = torch.tensor(B_np).type(torch.bool).cuda()
  elif dtype == "FP64": # DMMA(FP64), only supported on A100
    A_torch = torch.tensor(A_np).type(torch.float64).cuda()
    B_torch = torch.tensor(B_np).type(torch.float64).cuda()

  global RUN_NUMBER
  number, repeats = RUN_NUMBER

  for i in range(repeats):
      time_record = []
      for j in range(number):
          torch.cuda.synchronize()
          start = torch.cuda.Event(enable_timing=True)
          end = torch.cuda.Event(enable_timing=True)
          start.record()

          C_torch = torch.matmul(A_torch, B_torch)

          end.record()
          torch.cuda.synchronize()
          total = start.elapsed_time(end)
          time_record.append(total)
      if i == repeats - 1:
        print("Average bmm latency", np.mean(time_record))
        print("Median  bmm latency", np.median(time_record))
  print("BMM, B = %d H = %d, W = %d, K = %d" % (B, H, W, K))
  print("dtype = %s, A: %s, B: %s, C:%s" % (dtype, A_torch.dtype, B_torch.dtype, C_torch.dtype))
  print("------------------")

test_shapes = [
             [12, 512, 64, 512],
             [12, 512, 512, 64],
             [16, 512, 64, 512],
             [16, 512, 512, 64],
             [96, 2048, 128, 2048],
             [96, 2048, 2048, 128]
        ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--enable_cudnn", action="store_true")
    parser.add_argument("--number", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["FP16", "FP32", "TF32", "FP64", "BF16", "INT8", "BOOL"],
        default="FP16",
    )

    args = parser.parse_args()

    if args.enable_cudnn:
        assert torch.backends.cudnn.is_available()
        torch.backends.cudnn.enabled = True
    else:
        torch.backends.cudnn.enabled = False

    RUN_NUMBER = (args.number, args.repeats)

    # H, W, K for [H, K] * [K, W]
    for shape in test_shapes:
        B, H, K, W = shape
        bmm(B, H, W, K, args.dtype)

    print("cudnn: %s" % ("enabled" if torch.backends.cudnn.enabled else "disabled"))
