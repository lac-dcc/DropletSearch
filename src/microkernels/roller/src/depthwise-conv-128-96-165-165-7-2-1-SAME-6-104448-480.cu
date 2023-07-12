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
extern "C" __global__ void __launch_bounds__(480) default_function_kernel(float* __restrict__ compute, float* __restrict__ data, float* __restrict__ kernel) {
  float DepthwiseConv2d_local[2];
  __shared__ float PaddedInput_shared[5130];
  __shared__ float compute_shared[49];
  float PaddedInput_shared_local[2];
  float compute_shared_local[1];
  DepthwiseConv2d_local[0] = 0.000000e+00f;
  DepthwiseConv2d_local[1] = 0.000000e+00f;
  PaddedInput_shared[((int)threadIdx.x)] = ((((3 <= (((((int)blockIdx.x) % 17) * 10) + (((int)threadIdx.x) / 171))) && (3 <= (((int)threadIdx.x) % 171))) && ((((int)threadIdx.x) % 171) < 168)) ? data[(((((((((int)blockIdx.x) / 1632) * 5227200) + (((((int)blockIdx.x) % 1632) / 17) * 27225)) + ((((int)blockIdx.x) % 17) * 1650)) + ((((int)threadIdx.x) / 171) * 165)) + (((int)threadIdx.x) % 171)) - 498)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 480)] = ((((3 <= (((((int)blockIdx.x) % 17) * 10) + ((((int)threadIdx.x) + 480) / 171))) && (1 <= (((((int)threadIdx.x) / 3) + 46) % 57))) && (((((int)threadIdx.x) + 138) % 171) < 168)) ? data[(((((((((int)blockIdx.x) / 1632) * 5227200) + (((((int)blockIdx.x) % 1632) / 17) * 27225)) + ((((int)blockIdx.x) % 17) * 1650)) + (((((int)threadIdx.x) + 480) / 171) * 165)) + ((((int)threadIdx.x) + 138) % 171)) - 498)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 960)] = (((((((((int)blockIdx.x) % 17) * 5) + ((((int)threadIdx.x) + 960) / 342)) < 84) && (1 <= (((((int)threadIdx.x) / 3) + 35) % 57))) && (((((int)threadIdx.x) + 105) % 171) < 168)) ? data[(((((((((int)blockIdx.x) / 1632) * 5227200) + (((((int)blockIdx.x) % 1632) / 17) * 27225)) + ((((int)blockIdx.x) % 17) * 1650)) + (((((int)threadIdx.x) + 960) / 171) * 165)) + ((((int)threadIdx.x) + 105) % 171)) - 498)] : 0.000000e+00f);
  if ((((((int)blockIdx.x) % 17) * 10) + ((((int)threadIdx.x) + 1440) / 171)) < 171) {
    PaddedInput_shared[(((int)threadIdx.x) + 1440)] = (((((((((int)blockIdx.x) % 17) * 5) + ((((int)threadIdx.x) + 1440) / 342)) < 84) && (1 <= (((((int)threadIdx.x) / 3) + 24) % 57))) && (((((int)threadIdx.x) + 72) % 171) < 168)) ? data[(((((((((int)blockIdx.x) / 1632) * 5227200) + (((((int)blockIdx.x) % 1632) / 17) * 27225)) + ((((int)blockIdx.x) % 17) * 1650)) + (((((int)threadIdx.x) + 1440) / 171) * 165)) + ((((int)threadIdx.x) + 72) % 171)) - 498)] : 0.000000e+00f);
  }
  if ((((((int)blockIdx.x) % 17) * 10) + ((((int)threadIdx.x) + 1920) / 171)) < 171) {
    PaddedInput_shared[(((int)threadIdx.x) + 1920)] = (((((((((int)blockIdx.x) % 17) * 5) + ((((int)threadIdx.x) + 1920) / 342)) < 84) && (1 <= (((((int)threadIdx.x) / 3) + 13) % 57))) && (((((int)threadIdx.x) + 39) % 171) < 168)) ? data[(((((((((int)blockIdx.x) / 1632) * 5227200) + (((((int)blockIdx.x) % 1632) / 17) * 27225)) + ((((int)blockIdx.x) % 17) * 1650)) + (((((int)threadIdx.x) + 1920) / 171) * 165)) + ((((int)threadIdx.x) + 39) % 171)) - 498)] : 0.000000e+00f);
  }
  if ((((((int)blockIdx.x) % 17) * 10) + ((((((int)threadIdx.x) / 3) + 800) % 855) / 57)) < 171) {
    PaddedInput_shared[(((int)threadIdx.x) + 2400)] = (((((3 <= (((((int)blockIdx.x) % 17) * 10) + ((((((int)threadIdx.x) / 3) + 800) % 855) / 57))) && ((((((int)blockIdx.x) % 17) * 5) + ((((((int)threadIdx.x) / 3) + 800) % 855) / 114)) < 84)) && (1 <= (((((int)threadIdx.x) / 3) + 2) % 57))) && (((((int)threadIdx.x) + 6) % 171) < 168)) ? data[((((((((((int)blockIdx.x) / 1632) * 5227200) + (((((int)threadIdx.x) + 2400) / 2565) * 2613600)) + (((((int)blockIdx.x) % 1632) / 17) * 27225)) + ((((int)blockIdx.x) % 17) * 1650)) + (((((((int)threadIdx.x) / 3) + 800) % 855) / 57) * 165)) + ((((int)threadIdx.x) + 6) % 171)) - 498)] : 0.000000e+00f);
  }
  PaddedInput_shared[(((int)threadIdx.x) + 2880)] = ((((3 <= (((((int)blockIdx.x) % 17) * 10) + ((((int)threadIdx.x) + 315) / 171))) && (1 <= (((((int)threadIdx.x) / 3) + 48) % 57))) && (((((int)threadIdx.x) + 144) % 171) < 168)) ? data[((((((((((int)blockIdx.x) / 1632) * 5227200) + (((((int)threadIdx.x) + 2880) / 2565) * 2613600)) + (((((int)blockIdx.x) % 1632) / 17) * 27225)) + ((((int)blockIdx.x) % 17) * 1650)) + (((((int)threadIdx.x) + 315) / 171) * 165)) + ((((int)threadIdx.x) + 144) % 171)) - 498)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3360)] = (((1 <= (((((int)threadIdx.x) / 3) + 37) % 57)) && (((((int)threadIdx.x) + 111) % 171) < 168)) ? data[((((((((((int)blockIdx.x) / 1632) * 5227200) + (((((int)threadIdx.x) + 3360) / 2565) * 2613600)) + (((((int)blockIdx.x) % 1632) / 17) * 27225)) + ((((int)blockIdx.x) % 17) * 1650)) + (((((int)threadIdx.x) + 795) / 171) * 165)) + ((((int)threadIdx.x) + 111) % 171)) - 498)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3840)] = (((((((((int)blockIdx.x) % 17) * 5) + ((((int)threadIdx.x) + 1275) / 342)) < 84) && (1 <= (((((int)threadIdx.x) / 3) + 26) % 57))) && (((((int)threadIdx.x) + 78) % 171) < 168)) ? data[((((((((((int)blockIdx.x) / 1632) * 5227200) + (((((int)threadIdx.x) + 3840) / 2565) * 2613600)) + (((((int)blockIdx.x) % 1632) / 17) * 27225)) + ((((int)blockIdx.x) % 17) * 1650)) + (((((int)threadIdx.x) + 1275) / 171) * 165)) + ((((int)threadIdx.x) + 78) % 171)) - 498)] : 0.000000e+00f);
  if ((((((int)blockIdx.x) % 17) * 10) + ((((int)threadIdx.x) + 1755) / 171)) < 171) {
    PaddedInput_shared[(((int)threadIdx.x) + 4320)] = (((((((((int)blockIdx.x) % 17) * 5) + ((((int)threadIdx.x) + 1755) / 342)) < 84) && (1 <= (((((int)threadIdx.x) / 3) + 15) % 57))) && (((((int)threadIdx.x) + 45) % 171) < 168)) ? data[((((((((((int)blockIdx.x) / 1632) * 5227200) + (((((int)threadIdx.x) + 4320) / 2565) * 2613600)) + (((((int)blockIdx.x) % 1632) / 17) * 27225)) + ((((int)blockIdx.x) % 17) * 1650)) + (((((int)threadIdx.x) + 1755) / 171) * 165)) + ((((int)threadIdx.x) + 45) % 171)) - 498)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 330) {
    if ((((((int)blockIdx.x) % 17) * 10) + ((((int)threadIdx.x) + 2235) / 171)) < 171) {
      PaddedInput_shared[(((int)threadIdx.x) + 4800)] = (((((((((int)blockIdx.x) % 17) * 5) + ((((int)threadIdx.x) + 2235) / 342)) < 84) && (1 <= (((((int)threadIdx.x) / 3) + 4) % 57))) && (((((int)threadIdx.x) + 12) % 171) < 168)) ? data[((((((((((int)blockIdx.x) / 1632) * 5227200) + (((((int)threadIdx.x) + 4800) / 2565) * 2613600)) + (((((int)blockIdx.x) % 1632) / 17) * 27225)) + ((((int)blockIdx.x) % 17) * 1650)) + (((((int)threadIdx.x) + 2235) / 171) * 165)) + ((((int)threadIdx.x) + 12) % 171)) - 498)] : 0.000000e+00f);
    }
  }
  if (((int)threadIdx.x) < 49) {
    compute_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) % 1632) / 17) * 49) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int k_inner_outer = 0; k_inner_outer < 49; ++k_inner_outer) {
    if (((((((int)blockIdx.x) % 17) * 10) + (((((int)threadIdx.x) % 240) / 48) * 2)) + (k_inner_outer / 7)) < 171) {
      PaddedInput_shared_local[0] = PaddedInput_shared[((((((((int)threadIdx.x) / 240) * 2565) + (((((int)threadIdx.x) % 240) / 48) * 342)) + ((k_inner_outer / 7) * 171)) + ((((int)threadIdx.x) % 48) * 2)) + (k_inner_outer % 7))];
      if ((((((int)threadIdx.x) % 48) * 2) + (k_inner_outer % 7)) < 75) {
        PaddedInput_shared_local[1] = PaddedInput_shared[(((((((((int)threadIdx.x) / 240) * 2565) + (((((int)threadIdx.x) % 240) / 48) * 342)) + ((k_inner_outer / 7) * 171)) + ((((int)threadIdx.x) % 48) * 2)) + (k_inner_outer % 7)) + 96)];
      }
    }
    compute_shared_local[0] = compute_shared[k_inner_outer];
    if ((((((int)blockIdx.x) % 17) * 5) + ((((int)threadIdx.x) % 240) / 48)) < 83) {
      DepthwiseConv2d_local[0] = (DepthwiseConv2d_local[0] + (PaddedInput_shared_local[0] * compute_shared_local[0]));
      if ((((int)threadIdx.x) % 48) < 35) {
        DepthwiseConv2d_local[1] = (DepthwiseConv2d_local[1] + (PaddedInput_shared_local[1] * compute_shared_local[0]));
      }
    }
  }
  if ((((((int)blockIdx.x) % 17) * 5) + ((((int)threadIdx.x) % 240) / 48)) < 83) {
    compute[(((((((((int)blockIdx.x) / 1632) * 1322688) + ((((int)threadIdx.x) / 240) * 661344)) + (((((int)blockIdx.x) % 1632) / 17) * 6889)) + ((((int)blockIdx.x) % 17) * 415)) + (((((int)threadIdx.x) % 240) / 48) * 83)) + (((int)threadIdx.x) % 48))] = DepthwiseConv2d_local[0];
    if ((((int)threadIdx.x) % 48) < 35) {
      compute[((((((((((int)blockIdx.x) / 1632) * 1322688) + ((((int)threadIdx.x) / 240) * 661344)) + (((((int)blockIdx.x) % 1632) / 17) * 6889)) + ((((int)blockIdx.x) % 17) * 415)) + (((((int)threadIdx.x) % 240) / 48) * 83)) + (((int)threadIdx.x) % 48)) + 48)] = DepthwiseConv2d_local[1];
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

    dim3 grid(104448, 1, 1);
    dim3 block(480, 1, 1);
    for (int i = 0; i < 10; ++i)
    {
        default_function_kernel0<<<grid, block>>>((float*)Ad, (float*)Bd, (float*)Cd);
        cudaDeviceSynchronize();
    }
}
