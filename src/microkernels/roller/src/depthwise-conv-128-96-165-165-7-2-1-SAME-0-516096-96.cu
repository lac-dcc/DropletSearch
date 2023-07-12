#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "cu_helper.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <string>

int N = 128;
int C = 96;
std::string P = "SAME";
int S_height = 2, S_width = 2;
int NH = 165, KH = 7;
int NW = 165, KW = 7;


#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(96) default_function_kernel(float* __restrict__ compute, float* __restrict__ data, float* __restrict__ kernel) {
  float DepthwiseConv2d_local[2];
  __shared__ float PaddedInput_shared[1539];
  __shared__ float compute_shared[49];
  float PaddedInput_shared_local[2];
  float compute_shared_local[1];
  DepthwiseConv2d_local[0] = 0.000000e+00f;
  DepthwiseConv2d_local[1] = 0.000000e+00f;
  PaddedInput_shared[((int)threadIdx.x)] = (((0 < (((int)blockIdx.x) % 42)) && (3 <= ((int)threadIdx.x))) ? data[(((((((int)blockIdx.x) / 42) * 27225) + ((((int)blockIdx.x) % 42) * 660)) + ((int)threadIdx.x)) - 498)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 96)] = ((((3 <= (((((int)blockIdx.x) % 42) * 4) + ((((int)threadIdx.x) + 96) / 171))) && (1 <= (((((int)threadIdx.x) / 3) + 32) % 57))) && (((((int)threadIdx.x) + 96) % 171) < 168)) ? data[((((((((int)blockIdx.x) / 42) * 27225) + ((((int)blockIdx.x) % 42) * 660)) + (((((int)threadIdx.x) + 96) / 171) * 165)) + ((((int)threadIdx.x) + 96) % 171)) - 498)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 192)] = ((3 <= (((((int)blockIdx.x) % 42) * 4) + ((((int)threadIdx.x) + 192) / 171))) ? data[((((((((int)blockIdx.x) / 42) * 27225) + ((((int)blockIdx.x) % 42) * 660)) + (((((int)threadIdx.x) + 192) / 171) * 165)) + ((int)threadIdx.x)) - 477)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 288)] = ((((3 <= (((((int)blockIdx.x) % 42) * 4) + ((((int)threadIdx.x) + 288) / 171))) && (1 <= (((((int)threadIdx.x) / 3) + 39) % 57))) && (((((int)threadIdx.x) + 117) % 171) < 168)) ? data[((((((((int)blockIdx.x) / 42) * 27225) + ((((int)blockIdx.x) % 42) * 660)) + (((((int)threadIdx.x) + 288) / 171) * 165)) + ((((int)threadIdx.x) + 117) % 171)) - 498)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 384)] = ((3 <= (((((int)blockIdx.x) % 42) * 4) + ((((int)threadIdx.x) + 384) / 171))) ? data[((((((((int)blockIdx.x) / 42) * 27225) + ((((int)blockIdx.x) % 42) * 660)) + (((((int)threadIdx.x) + 384) / 171) * 165)) + ((int)threadIdx.x)) - 456)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 480)] = ((((3 <= (((((int)blockIdx.x) % 42) * 4) + ((((int)threadIdx.x) + 480) / 171))) && (1 <= (((((int)threadIdx.x) / 3) + 46) % 57))) && (((((int)threadIdx.x) + 138) % 171) < 168)) ? data[((((((((int)blockIdx.x) / 42) * 27225) + ((((int)blockIdx.x) % 42) * 660)) + (((((int)threadIdx.x) + 480) / 171) * 165)) + ((((int)threadIdx.x) + 138) % 171)) - 498)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 576)] = data[((((((((int)blockIdx.x) / 42) * 27225) + ((((int)blockIdx.x) % 42) * 660)) + (((((int)threadIdx.x) + 576) / 171) * 165)) + ((int)threadIdx.x)) - 435)];
  PaddedInput_shared[(((int)threadIdx.x) + 672)] = (((((((((int)threadIdx.x) + 672) / 684) + (((int)blockIdx.x) % 42)) < 42) && (1 <= (((((int)threadIdx.x) / 3) + 53) % 57))) && (((((int)threadIdx.x) + 159) % 171) < 168)) ? data[((((((((int)blockIdx.x) / 42) * 27225) + ((((int)blockIdx.x) % 42) * 660)) + (((((int)threadIdx.x) + 672) / 171) * 165)) + ((((int)threadIdx.x) + 159) % 171)) - 498)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 768)] = (((((((((int)threadIdx.x) + 768) / 684) + (((int)blockIdx.x) % 42)) < 42) && (1 <= (((((int)threadIdx.x) / 3) + 28) % 57))) && (((((int)threadIdx.x) + 84) % 171) < 168)) ? data[((((((((int)blockIdx.x) / 42) * 27225) + ((((int)blockIdx.x) % 42) * 660)) + (((((int)threadIdx.x) + 768) / 171) * 165)) + ((((int)threadIdx.x) + 84) % 171)) - 498)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 864)] = (((((((int)threadIdx.x) + 864) / 684) + (((int)blockIdx.x) % 42)) < 42) ? data[((((((((int)blockIdx.x) / 42) * 27225) + ((((int)blockIdx.x) % 42) * 660)) + (((((int)threadIdx.x) + 864) / 171) * 165)) + ((int)threadIdx.x)) - 489)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 960)] = (((((((((int)threadIdx.x) + 960) / 684) + (((int)blockIdx.x) % 42)) < 42) && (1 <= (((((int)threadIdx.x) / 3) + 35) % 57))) && (((((int)threadIdx.x) + 105) % 171) < 168)) ? data[((((((((int)blockIdx.x) / 42) * 27225) + ((((int)blockIdx.x) % 42) * 660)) + (((((int)threadIdx.x) + 960) / 171) * 165)) + ((((int)threadIdx.x) + 105) % 171)) - 498)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1056)] = (((((((int)threadIdx.x) + 1056) / 684) + (((int)blockIdx.x) % 42)) < 42) ? data[((((((((int)blockIdx.x) / 42) * 27225) + ((((int)blockIdx.x) % 42) * 660)) + (((((int)threadIdx.x) + 1056) / 171) * 165)) + ((int)threadIdx.x)) - 468)] : 0.000000e+00f);
  if ((((((int)blockIdx.x) % 42) * 4) + ((((int)threadIdx.x) + 1152) / 171)) < 171) {
    PaddedInput_shared[(((int)threadIdx.x) + 1152)] = (((((((((int)threadIdx.x) + 1152) / 684) + (((int)blockIdx.x) % 42)) < 42) && (1 <= (((((int)threadIdx.x) / 3) + 42) % 57))) && (((((int)threadIdx.x) + 126) % 171) < 168)) ? data[((((((((int)blockIdx.x) / 42) * 27225) + ((((int)blockIdx.x) % 42) * 660)) + (((((int)threadIdx.x) + 1152) / 171) * 165)) + ((((int)threadIdx.x) + 126) % 171)) - 498)] : 0.000000e+00f);
  }
  if ((((((int)blockIdx.x) % 42) * 4) + ((((int)threadIdx.x) + 1248) / 171)) < 171) {
    PaddedInput_shared[(((int)threadIdx.x) + 1248)] = (((((((int)threadIdx.x) + 1248) / 684) + (((int)blockIdx.x) % 42)) < 42) ? data[((((((((int)blockIdx.x) / 42) * 27225) + ((((int)blockIdx.x) % 42) * 660)) + (((((int)threadIdx.x) + 1248) / 171) * 165)) + ((int)threadIdx.x)) - 447)] : 0.000000e+00f);
  }
  if ((((((int)blockIdx.x) % 42) * 4) + ((((int)threadIdx.x) + 1344) / 171)) < 171) {
    PaddedInput_shared[(((int)threadIdx.x) + 1344)] = (((((((((int)threadIdx.x) + 1344) / 684) + (((int)blockIdx.x) % 42)) < 42) && (1 <= (((((int)threadIdx.x) / 3) + 49) % 57))) && (((((int)threadIdx.x) + 147) % 171) < 168)) ? data[((((((((int)blockIdx.x) / 42) * 27225) + ((((int)blockIdx.x) % 42) * 660)) + (((((int)threadIdx.x) + 1344) / 171) * 165)) + ((((int)threadIdx.x) + 147) % 171)) - 498)] : 0.000000e+00f);
  }
  if ((((((int)blockIdx.x) % 42) * 4) + ((((int)threadIdx.x) + 1440) / 171)) < 171) {
    PaddedInput_shared[(((int)threadIdx.x) + 1440)] = (((((((int)threadIdx.x) + 1440) / 684) + (((int)blockIdx.x) % 42)) < 42) ? data[((((((((int)blockIdx.x) / 42) * 27225) + ((((int)blockIdx.x) % 42) * 660)) + (((((int)threadIdx.x) + 1440) / 171) * 165)) + ((int)threadIdx.x)) - 426)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 3) {
    if ((((int)blockIdx.x) % 42) < 41) {
      PaddedInput_shared[(((int)threadIdx.x) + 1536)] = 0.000000e+00f;
    }
  }
  if (((int)threadIdx.x) < 49) {
    compute_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) % 4032) / 42) * 49) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int k_inner_outer = 0; k_inner_outer < 49; ++k_inner_outer) {
    if (((((((int)blockIdx.x) % 42) * 4) + ((((int)threadIdx.x) / 48) * 2)) + (k_inner_outer / 7)) < 171) {
      PaddedInput_shared_local[0] = PaddedInput_shared[(((((((int)threadIdx.x) / 48) * 342) + ((k_inner_outer / 7) * 171)) + ((((int)threadIdx.x) % 48) * 2)) + (k_inner_outer % 7))];
      if ((((((int)threadIdx.x) % 48) * 2) + (k_inner_outer % 7)) < 75) {
        PaddedInput_shared_local[1] = PaddedInput_shared[((((((((int)threadIdx.x) / 48) * 342) + ((k_inner_outer / 7) * 171)) + ((((int)threadIdx.x) % 48) * 2)) + (k_inner_outer % 7)) + 96)];
      }
    }
    compute_shared_local[0] = compute_shared[k_inner_outer];
    if ((((((int)blockIdx.x) % 42) * 2) + (((int)threadIdx.x) / 48)) < 83) {
      DepthwiseConv2d_local[0] = (DepthwiseConv2d_local[0] + (PaddedInput_shared_local[0] * compute_shared_local[0]));
      if ((((int)threadIdx.x) % 48) < 35) {
        DepthwiseConv2d_local[1] = (DepthwiseConv2d_local[1] + (PaddedInput_shared_local[1] * compute_shared_local[0]));
      }
    }
  }
  if ((((((int)blockIdx.x) % 42) * 2) + (((int)threadIdx.x) / 48)) < 83) {
    compute[(((((((int)blockIdx.x) / 42) * 6889) + ((((int)blockIdx.x) % 42) * 166)) + ((((int)threadIdx.x) / 48) * 83)) + (((int)threadIdx.x) % 48))] = DepthwiseConv2d_local[0];
    if ((((int)threadIdx.x) % 48) < 35) {
      compute[((((((((int)blockIdx.x) / 42) * 6889) + ((((int)blockIdx.x) % 42) * 166)) + ((((int)threadIdx.x) / 48) * 83)) + (((int)threadIdx.x) % 48)) + 48)] = DepthwiseConv2d_local[1];
    }
  }
}

int main(int argc, char *argv[])
{
    int input_size0 = N * C * (NH + KH - 1) * (NW + KW - 1);
    int input_size1 = C * KH * KW;
    int output_size;
   if (P == std::string("VALID")){
       output_size = N * C * ((NH - KH + 1) / S_height + 1) * ((NW - KW + 1) / S_width + 1);
   } else if (P == std::string("SAME")){
       output_size = N * C * (NH / S_height + 1) * (NW / S_width + 1);
   }

    checkCudaErrors(cuInit(0));
    CUdevice device;
    checkCudaErrors(cuDeviceGet(&device, 0));
    CUcontext context;
    checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO/*CU_CTX_SCHED_YIELD*/ | CU_CTX_MAP_HOST, device));

    float *Ah, *Bh;
    float *Ad, *Bd, *Cd;
    Ah = (float*)malloc(input_size0 * sizeof(float));
    Bh = (float*)malloc(input_size1 * sizeof(float));

    cudaMalloc((void **)&Ad, input_size0 * sizeof(float));
    cudaMalloc((void **)&Bd, input_size1 * sizeof(float));
    cudaMalloc((void **)&Cd, output_size * sizeof(float));

    srand(1);
    for (int i = 0; i < input_size0; ++ i)
        Ah[i] = 1;
    for (int i = 0; i < input_size1; ++ i)
        Bh[i] = 1;

    cudaMemcpy(Ad, Ah, input_size0 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, Bh, input_size1 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(516096, 1, 1);
    dim3 block(96, 1, 1);
    for (int i = 0; i < 10; ++i)
    {
        default_function_kernel0<<<grid, block>>>((float*)Ad, (float*)Bd, (float*)Cd);
        cudaDeviceSynchronize();
    }
}
