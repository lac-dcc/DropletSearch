#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "cu_helper.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <string>

int M = 65536, N = 1024;


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
extern "C" __global__ void __launch_bounds__(512) default_function_kernel(float* __restrict__ A, float* __restrict__ compute) {
  float compute_local[1];
  __shared__ float A_shared[4096];
  float A_shared_local[1];
  compute_local[0] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 128; ++k_outer) {
    __syncthreads();
    A_shared[((int)threadIdx.x)] = A[((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_outer * 8)) + (((int)threadIdx.x) & 7))];
    A_shared[(((int)threadIdx.x) + 512)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_outer * 8)) + (((int)threadIdx.x) & 7)) + 65536)];
    A_shared[(((int)threadIdx.x) + 1024)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_outer * 8)) + (((int)threadIdx.x) & 7)) + 131072)];
    A_shared[(((int)threadIdx.x) + 1536)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_outer * 8)) + (((int)threadIdx.x) & 7)) + 196608)];
    A_shared[(((int)threadIdx.x) + 2048)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_outer * 8)) + (((int)threadIdx.x) & 7)) + 262144)];
    A_shared[(((int)threadIdx.x) + 2560)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_outer * 8)) + (((int)threadIdx.x) & 7)) + 327680)];
    A_shared[(((int)threadIdx.x) + 3072)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_outer * 8)) + (((int)threadIdx.x) & 7)) + 393216)];
    A_shared[(((int)threadIdx.x) + 3584)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_outer * 8)) + (((int)threadIdx.x) & 7)) + 458752)];
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 8; ++k_inner_outer) {
      A_shared_local[0] = A_shared[((((int)threadIdx.x) * 8) + k_inner_outer)];
      compute_local[0] = (compute_local[0] + A_shared_local[0]);
    }
  }
  compute[((((int)blockIdx.x) * 512) + ((int)threadIdx.x))] = compute_local[0];
}


int main(int argc, char *argv[])
{
    std::string path;
    int input_size = M * N;
    int output_size = M;

    checkCudaErrors(cuInit(0));
    CUdevice device;
    checkCudaErrors(cuDeviceGet(&device, 0));
    CUcontext context;
    checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO/*CU_CTX_SCHED_YIELD*/ | CU_CTX_MAP_HOST, device));

    float *Ah, *Ch;
    float *Ad, *Cd;
    Ah = (float*)malloc(input_size * sizeof(float));
    Ch = (float*)malloc(output_size * sizeof(float));

    cudaMalloc((void **)&Ad, input_size * sizeof(float));
    cudaMalloc((void **)&Cd, output_size * sizeof(float));

    srand(1);
    for (int i = 0; i < input_size; ++ i)
        Ah[i] = 1;

    cudaMemcpy(Ad, Ah, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Cd, Ch, output_size * sizeof(float), cudaMemcpyHostToDevice);

    int grid_size = 128;
    int block_size = 512;
    dim3 grid(grid_size, 1, 1);
    dim3 block(block_size, 1, 1);

    int numBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, default_function_kernel0, block_size, 0);
    fprintf(stderr, "Active blocks per SM = %d\n", numBlocks);
 
    for (int i = 0; i < 10; ++i)
    {
        default_function_kernel0<<<grid, block>>>((float*)Ad, (float*)Cd);
        cudaDeviceSynchronize();
    }
}
