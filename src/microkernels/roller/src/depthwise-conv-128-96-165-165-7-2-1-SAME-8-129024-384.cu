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
extern "C" __global__ void __launch_bounds__(384) default_function_kernel(float* __restrict__ compute, float* __restrict__ data, float* __restrict__ kernel) {
  float DepthwiseConv2d_local[2];
  __shared__ float PaddedInput_shared[4446];
  __shared__ float compute_shared[49];
  float PaddedInput_shared_local[2];
  float compute_shared_local[1];
  DepthwiseConv2d_local[0] = 0.000000e+00f;
  DepthwiseConv2d_local[1] = 0.000000e+00f;
  PaddedInput_shared[((int)threadIdx.x)] = ((((3 <= (((((int)blockIdx.x) % 21) * 8) + (((int)threadIdx.x) / 171))) && (3 <= (((int)threadIdx.x) % 171))) && ((((int)threadIdx.x) % 171) < 168)) ? data[(((((((((int)blockIdx.x) / 2016) * 5227200) + (((((int)blockIdx.x) % 2016) / 21) * 27225)) + ((((int)blockIdx.x) % 21) * 1320)) + ((((int)threadIdx.x) / 171) * 165)) + (((int)threadIdx.x) % 171)) - 498)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 384)] = ((((3 <= (((((int)blockIdx.x) % 21) * 8) + ((((int)threadIdx.x) + 384) / 171))) && (1 <= (((((int)threadIdx.x) / 3) + 14) % 57))) && (((((int)threadIdx.x) + 42) % 171) < 168)) ? data[(((((((((int)blockIdx.x) / 2016) * 5227200) + (((((int)blockIdx.x) % 2016) / 21) * 27225)) + ((((int)blockIdx.x) % 21) * 1320)) + (((((int)threadIdx.x) + 384) / 171) * 165)) + ((((int)threadIdx.x) + 42) % 171)) - 498)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 768)] = (((1 <= (((((int)threadIdx.x) / 3) + 28) % 57)) && (((((int)threadIdx.x) + 84) % 171) < 168)) ? data[(((((((((int)blockIdx.x) / 2016) * 5227200) + (((((int)blockIdx.x) % 2016) / 21) * 27225)) + ((((int)blockIdx.x) % 21) * 1320)) + (((((int)threadIdx.x) + 768) / 171) * 165)) + ((((int)threadIdx.x) + 84) % 171)) - 498)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 1152)] = (((((((((int)threadIdx.x) + 1152) / 1368) + (((int)blockIdx.x) % 21)) < 21) && (1 <= (((((int)threadIdx.x) / 3) + 42) % 57))) && (((((int)threadIdx.x) + 126) % 171) < 168)) ? data[(((((((((int)blockIdx.x) / 2016) * 5227200) + (((((int)blockIdx.x) % 2016) / 21) * 27225)) + ((((int)blockIdx.x) % 21) * 1320)) + (((((int)threadIdx.x) + 1152) / 171) * 165)) + ((((int)threadIdx.x) + 126) % 171)) - 498)] : 0.000000e+00f);
  if ((((((int)blockIdx.x) % 21) * 8) + ((((int)threadIdx.x) + 1536) / 171)) < 171) {
    PaddedInput_shared[(((int)threadIdx.x) + 1536)] = (((((((((int)threadIdx.x) + 1536) / 1368) + (((int)blockIdx.x) % 21)) < 21) && (1 <= (((((int)threadIdx.x) / 3) + 56) % 57))) && (((((int)threadIdx.x) + 168) % 171) < 168)) ? data[(((((((((int)blockIdx.x) / 2016) * 5227200) + (((((int)blockIdx.x) % 2016) / 21) * 27225)) + ((((int)blockIdx.x) % 21) * 1320)) + (((((int)threadIdx.x) + 1536) / 171) * 165)) + ((((int)threadIdx.x) + 168) % 171)) - 498)] : 0.000000e+00f);
  }
  if ((((((int)blockIdx.x) % 21) * 8) + ((((((int)threadIdx.x) / 3) + 640) % 741) / 57)) < 171) {
    PaddedInput_shared[(((int)threadIdx.x) + 1920)] = (((((3 <= (((((int)blockIdx.x) % 21) * 8) + ((((((int)threadIdx.x) / 3) + 640) % 741) / 57))) && ((((((((int)threadIdx.x) / 3) + 640) % 741) / 456) + (((int)blockIdx.x) % 21)) < 21)) && (1 <= (((((int)threadIdx.x) / 3) + 13) % 57))) && (((((int)threadIdx.x) + 39) % 171) < 168)) ? data[((((((((((int)blockIdx.x) / 2016) * 5227200) + (((((int)threadIdx.x) + 1920) / 2223) * 2613600)) + (((((int)blockIdx.x) % 2016) / 21) * 27225)) + ((((int)blockIdx.x) % 21) * 1320)) + (((((((int)threadIdx.x) / 3) + 640) % 741) / 57) * 165)) + ((((int)threadIdx.x) + 39) % 171)) - 498)] : 0.000000e+00f);
  }
  PaddedInput_shared[(((int)threadIdx.x) + 2304)] = ((((3 <= (((((int)blockIdx.x) % 21) * 8) + ((((int)threadIdx.x) + 81) / 171))) && (1 <= (((((int)threadIdx.x) / 3) + 27) % 57))) && (((((int)threadIdx.x) + 81) % 171) < 168)) ? data[((((((((((int)blockIdx.x) / 2016) * 5227200) + (((((int)threadIdx.x) + 2304) / 2223) * 2613600)) + (((((int)blockIdx.x) % 2016) / 21) * 27225)) + ((((int)blockIdx.x) % 21) * 1320)) + (((((int)threadIdx.x) + 81) / 171) * 165)) + ((((int)threadIdx.x) + 81) % 171)) - 498)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 2688)] = ((((3 <= (((((int)blockIdx.x) % 21) * 8) + ((((int)threadIdx.x) + 465) / 171))) && (1 <= (((((int)threadIdx.x) / 3) + 41) % 57))) && (((((int)threadIdx.x) + 123) % 171) < 168)) ? data[((((((((((int)blockIdx.x) / 2016) * 5227200) + (((((int)threadIdx.x) + 2688) / 2223) * 2613600)) + (((((int)blockIdx.x) % 2016) / 21) * 27225)) + ((((int)blockIdx.x) % 21) * 1320)) + (((((int)threadIdx.x) + 465) / 171) * 165)) + ((((int)threadIdx.x) + 123) % 171)) - 498)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3072)] = (((1 <= (((((int)threadIdx.x) / 3) + 55) % 57)) && (((((int)threadIdx.x) + 165) % 171) < 168)) ? data[((((((((((int)blockIdx.x) / 2016) * 5227200) + (((((int)threadIdx.x) + 3072) / 2223) * 2613600)) + (((((int)blockIdx.x) % 2016) / 21) * 27225)) + ((((int)blockIdx.x) % 21) * 1320)) + (((((int)threadIdx.x) + 849) / 171) * 165)) + ((((int)threadIdx.x) + 165) % 171)) - 498)] : 0.000000e+00f);
  PaddedInput_shared[(((int)threadIdx.x) + 3456)] = (((((((((int)threadIdx.x) + 1233) / 1368) + (((int)blockIdx.x) % 21)) < 21) && (1 <= (((((int)threadIdx.x) / 3) + 12) % 57))) && (((((int)threadIdx.x) + 36) % 171) < 168)) ? data[((((((((((int)blockIdx.x) / 2016) * 5227200) + (((((int)threadIdx.x) + 3456) / 2223) * 2613600)) + (((((int)blockIdx.x) % 2016) / 21) * 27225)) + ((((int)blockIdx.x) % 21) * 1320)) + (((((int)threadIdx.x) + 1233) / 171) * 165)) + ((((int)threadIdx.x) + 36) % 171)) - 498)] : 0.000000e+00f);
  if ((((((int)blockIdx.x) % 21) * 8) + ((((int)threadIdx.x) + 1617) / 171)) < 171) {
    PaddedInput_shared[(((int)threadIdx.x) + 3840)] = (((((((((int)threadIdx.x) + 1617) / 1368) + (((int)blockIdx.x) % 21)) < 21) && (1 <= (((((int)threadIdx.x) / 3) + 26) % 57))) && (((((int)threadIdx.x) + 78) % 171) < 168)) ? data[((((((((((int)blockIdx.x) / 2016) * 5227200) + (((((int)threadIdx.x) + 3840) / 2223) * 2613600)) + (((((int)blockIdx.x) % 2016) / 21) * 27225)) + ((((int)blockIdx.x) % 21) * 1320)) + (((((int)threadIdx.x) + 1617) / 171) * 165)) + ((((int)threadIdx.x) + 78) % 171)) - 498)] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 222) {
    if ((((((int)blockIdx.x) % 21) * 8) + ((((int)threadIdx.x) + 2001) / 171)) < 171) {
      PaddedInput_shared[(((int)threadIdx.x) + 4224)] = (((((((((int)threadIdx.x) + 2001) / 1368) + (((int)blockIdx.x) % 21)) < 21) && (1 <= (((((int)threadIdx.x) / 3) + 40) % 57))) && (((((int)threadIdx.x) + 120) % 171) < 168)) ? data[((((((((((int)blockIdx.x) / 2016) * 5227200) + (((((int)threadIdx.x) + 4224) / 2223) * 2613600)) + (((((int)blockIdx.x) % 2016) / 21) * 27225)) + ((((int)blockIdx.x) % 21) * 1320)) + (((((int)threadIdx.x) + 2001) / 171) * 165)) + ((((int)threadIdx.x) + 120) % 171)) - 498)] : 0.000000e+00f);
    }
  }
  if (((int)threadIdx.x) < 49) {
    compute_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) % 2016) / 21) * 49) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int k_inner_outer = 0; k_inner_outer < 49; ++k_inner_outer) {
    if (((((((int)blockIdx.x) % 21) * 8) + (((((int)threadIdx.x) % 192) / 48) * 2)) + (k_inner_outer / 7)) < 171) {
      PaddedInput_shared_local[0] = PaddedInput_shared[((((((((int)threadIdx.x) / 192) * 2223) + (((((int)threadIdx.x) % 192) / 48) * 342)) + ((k_inner_outer / 7) * 171)) + ((((int)threadIdx.x) % 48) * 2)) + (k_inner_outer % 7))];
      if ((((((int)threadIdx.x) % 48) * 2) + (k_inner_outer % 7)) < 75) {
        PaddedInput_shared_local[1] = PaddedInput_shared[(((((((((int)threadIdx.x) / 192) * 2223) + (((((int)threadIdx.x) % 192) / 48) * 342)) + ((k_inner_outer / 7) * 171)) + ((((int)threadIdx.x) % 48) * 2)) + (k_inner_outer % 7)) + 96)];
      }
    }
    compute_shared_local[0] = compute_shared[k_inner_outer];
    if ((((((int)blockIdx.x) % 21) * 4) + ((((int)threadIdx.x) % 192) / 48)) < 83) {
      DepthwiseConv2d_local[0] = (DepthwiseConv2d_local[0] + (PaddedInput_shared_local[0] * compute_shared_local[0]));
      if ((((int)threadIdx.x) % 48) < 35) {
        DepthwiseConv2d_local[1] = (DepthwiseConv2d_local[1] + (PaddedInput_shared_local[1] * compute_shared_local[0]));
      }
    }
  }
  if ((((((int)blockIdx.x) % 21) * 4) + ((((int)threadIdx.x) % 192) / 48)) < 83) {
    compute[(((((((((int)blockIdx.x) / 2016) * 1322688) + ((((int)threadIdx.x) / 192) * 661344)) + (((((int)blockIdx.x) % 2016) / 21) * 6889)) + ((((int)blockIdx.x) % 21) * 332)) + (((((int)threadIdx.x) % 192) / 48) * 83)) + (((int)threadIdx.x) % 48))] = DepthwiseConv2d_local[0];
    if ((((int)threadIdx.x) % 48) < 35) {
      compute[((((((((((int)blockIdx.x) / 2016) * 1322688) + ((((int)threadIdx.x) / 192) * 661344)) + (((((int)blockIdx.x) % 2016) / 21) * 6889)) + ((((int)blockIdx.x) % 21) * 332)) + (((((int)threadIdx.x) % 192) / 48) * 83)) + (((int)threadIdx.x) % 48)) + 48)] = DepthwiseConv2d_local[1];
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

    dim3 grid(129024, 1, 1);
    dim3 block(384, 1, 1);
    for (int i = 0; i < 10; ++i)
    {
        default_function_kernel0<<<grid, block>>>((float*)Ad, (float*)Bd, (float*)Cd);
        cudaDeviceSynchronize();
    }
}
