#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "cu_helper.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <string>

int N = 128;
int C = 672;
std::string P = "SAME";
int S_height = 2, S_width = 2;
int NH = 21, KH = 3;
int NW = 21, KW = 3;


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
extern "C" __global__ void __launch_bounds__(512) default_function_kernel(float* __restrict__ Pool2d, float* __restrict__ data) {
  if ((((int)threadIdx.x) & 127) < 88) {
    Pool2d[((((((int)blockIdx.x) * 484) + ((((int)threadIdx.x) >> 7) * 121)) + (((((int)threadIdx.x) & 127) >> 3) * 11)) + (((int)threadIdx.x) & 7))] = 0.000000e+00f;
    if ((((int)threadIdx.x) & 7) < 3) {
      Pool2d[(((((((int)blockIdx.x) * 484) + ((((int)threadIdx.x) >> 7) * 121)) + (((((int)threadIdx.x) & 127) >> 3) * 11)) + (((int)threadIdx.x) & 7)) + 8)] = 0.000000e+00f;
    }
  }
  for (int k_inner_outer = 0; k_inner_outer < 9; ++k_inner_outer) {
    if ((((int)threadIdx.x) & 127) < 88) {
      Pool2d[((((((int)blockIdx.x) * 484) + ((((int)threadIdx.x) >> 7) * 121)) + (((((int)threadIdx.x) & 127) >> 3) * 11)) + (((int)threadIdx.x) & 7))] = (Pool2d[((((((int)blockIdx.x) * 484) + ((((int)threadIdx.x) >> 7) * 121)) + (((((int)threadIdx.x) & 127) >> 3) * 11)) + (((int)threadIdx.x) & 7))] + (data[((((((((int)blockIdx.x) * 2116) + ((((int)threadIdx.x) >> 7) * 529)) + (((((int)threadIdx.x) & 127) >> 3) * 46)) + ((k_inner_outer / 3) * 23)) + ((((int)threadIdx.x) & 7) * 2)) + (k_inner_outer % 3))] * 1.111111e-01f));
      if ((((int)threadIdx.x) & 7) < 3) {
        Pool2d[(((((((int)blockIdx.x) * 484) + ((((int)threadIdx.x) >> 7) * 121)) + (((((int)threadIdx.x) & 127) >> 3) * 11)) + (((int)threadIdx.x) & 7)) + 8)] = (Pool2d[(((((((int)blockIdx.x) * 484) + ((((int)threadIdx.x) >> 7) * 121)) + (((((int)threadIdx.x) & 127) >> 3) * 11)) + (((int)threadIdx.x) & 7)) + 8)] + (data[(((((((((int)blockIdx.x) * 2116) + ((((int)threadIdx.x) >> 7) * 529)) + (((((int)threadIdx.x) & 127) >> 3) * 46)) + ((k_inner_outer / 3) * 23)) + ((((int)threadIdx.x) & 7) * 2)) + (k_inner_outer % 3)) + 16)] * 1.111111e-01f));
      }
    }
  }
}

int main(int argc, char *argv[])
{
    int input_size0 = N * C * (NH + KH - 1) * (NW + KW - 1);
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

    float *Ah;
    float *Ad, *Cd;
    Ah = (float*)malloc(input_size0 * sizeof(float));

    cudaMalloc((void **)&Ad, input_size0 * sizeof(float));
    cudaMalloc((void **)&Cd, output_size * sizeof(float));

    srand(1);
    for (int i = 0; i < input_size0; ++ i)
        Ah[i] = 1;

    cudaMemcpy(Ad, Ah, input_size0 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(21504, 1, 1);
    dim3 block(512, 1, 1);
    for (int i = 0; i < 10; ++i)
    {
        default_function_kernel0<<<grid, block>>>((float*)Cd, (float*)Ad);
        cudaDeviceSynchronize();
    }
}
