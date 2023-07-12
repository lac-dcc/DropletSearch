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
extern "C" __global__ void __launch_bounds__(128) default_function_kernel(float* __restrict__ A, float* __restrict__ compute) {
  float compute_local[4];
  __shared__ float A_shared[8192];
  float A_shared_local[4];
  compute_local[0] = 0.000000e+00f;
  compute_local[1] = 0.000000e+00f;
  compute_local[2] = 0.000000e+00f;
  compute_local[3] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 64; ++k_outer) {
    __syncthreads();
    A_shared[((int)threadIdx.x)] = A[((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15))];
    A_shared[(((int)threadIdx.x) + 128)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 8192)];
    A_shared[(((int)threadIdx.x) + 256)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 16384)];
    A_shared[(((int)threadIdx.x) + 384)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 24576)];
    A_shared[(((int)threadIdx.x) + 512)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 32768)];
    A_shared[(((int)threadIdx.x) + 640)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 40960)];
    A_shared[(((int)threadIdx.x) + 768)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 49152)];
    A_shared[(((int)threadIdx.x) + 896)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 57344)];
    A_shared[(((int)threadIdx.x) + 1024)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 65536)];
    A_shared[(((int)threadIdx.x) + 1152)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 73728)];
    A_shared[(((int)threadIdx.x) + 1280)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 81920)];
    A_shared[(((int)threadIdx.x) + 1408)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 90112)];
    A_shared[(((int)threadIdx.x) + 1536)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 98304)];
    A_shared[(((int)threadIdx.x) + 1664)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 106496)];
    A_shared[(((int)threadIdx.x) + 1792)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 114688)];
    A_shared[(((int)threadIdx.x) + 1920)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 122880)];
    A_shared[(((int)threadIdx.x) + 2048)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 131072)];
    A_shared[(((int)threadIdx.x) + 2176)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 139264)];
    A_shared[(((int)threadIdx.x) + 2304)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 147456)];
    A_shared[(((int)threadIdx.x) + 2432)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 155648)];
    A_shared[(((int)threadIdx.x) + 2560)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 163840)];
    A_shared[(((int)threadIdx.x) + 2688)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 172032)];
    A_shared[(((int)threadIdx.x) + 2816)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 180224)];
    A_shared[(((int)threadIdx.x) + 2944)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 188416)];
    A_shared[(((int)threadIdx.x) + 3072)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 196608)];
    A_shared[(((int)threadIdx.x) + 3200)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 204800)];
    A_shared[(((int)threadIdx.x) + 3328)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 212992)];
    A_shared[(((int)threadIdx.x) + 3456)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 221184)];
    A_shared[(((int)threadIdx.x) + 3584)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 229376)];
    A_shared[(((int)threadIdx.x) + 3712)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 237568)];
    A_shared[(((int)threadIdx.x) + 3840)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 245760)];
    A_shared[(((int)threadIdx.x) + 3968)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 253952)];
    A_shared[(((int)threadIdx.x) + 4096)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 262144)];
    A_shared[(((int)threadIdx.x) + 4224)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 270336)];
    A_shared[(((int)threadIdx.x) + 4352)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 278528)];
    A_shared[(((int)threadIdx.x) + 4480)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 286720)];
    A_shared[(((int)threadIdx.x) + 4608)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 294912)];
    A_shared[(((int)threadIdx.x) + 4736)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 303104)];
    A_shared[(((int)threadIdx.x) + 4864)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 311296)];
    A_shared[(((int)threadIdx.x) + 4992)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 319488)];
    A_shared[(((int)threadIdx.x) + 5120)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 327680)];
    A_shared[(((int)threadIdx.x) + 5248)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 335872)];
    A_shared[(((int)threadIdx.x) + 5376)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 344064)];
    A_shared[(((int)threadIdx.x) + 5504)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 352256)];
    A_shared[(((int)threadIdx.x) + 5632)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 360448)];
    A_shared[(((int)threadIdx.x) + 5760)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 368640)];
    A_shared[(((int)threadIdx.x) + 5888)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 376832)];
    A_shared[(((int)threadIdx.x) + 6016)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 385024)];
    A_shared[(((int)threadIdx.x) + 6144)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 393216)];
    A_shared[(((int)threadIdx.x) + 6272)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 401408)];
    A_shared[(((int)threadIdx.x) + 6400)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 409600)];
    A_shared[(((int)threadIdx.x) + 6528)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 417792)];
    A_shared[(((int)threadIdx.x) + 6656)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 425984)];
    A_shared[(((int)threadIdx.x) + 6784)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 434176)];
    A_shared[(((int)threadIdx.x) + 6912)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 442368)];
    A_shared[(((int)threadIdx.x) + 7040)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 450560)];
    A_shared[(((int)threadIdx.x) + 7168)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 458752)];
    A_shared[(((int)threadIdx.x) + 7296)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 466944)];
    A_shared[(((int)threadIdx.x) + 7424)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 475136)];
    A_shared[(((int)threadIdx.x) + 7552)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 483328)];
    A_shared[(((int)threadIdx.x) + 7680)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 491520)];
    A_shared[(((int)threadIdx.x) + 7808)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 499712)];
    A_shared[(((int)threadIdx.x) + 7936)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 507904)];
    A_shared[(((int)threadIdx.x) + 8064)] = A[(((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) & 15)) + 516096)];
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 16; ++k_inner_outer) {
      A_shared_local[0] = A_shared[((((int)threadIdx.x) * 16) + k_inner_outer)];
      A_shared_local[1] = A_shared[(((((int)threadIdx.x) * 16) + k_inner_outer) + 2048)];
      A_shared_local[2] = A_shared[(((((int)threadIdx.x) * 16) + k_inner_outer) + 4096)];
      A_shared_local[3] = A_shared[(((((int)threadIdx.x) * 16) + k_inner_outer) + 6144)];
      compute_local[0] = (compute_local[0] + A_shared_local[0]);
      compute_local[1] = (compute_local[1] + A_shared_local[1]);
      compute_local[2] = (compute_local[2] + A_shared_local[2]);
      compute_local[3] = (compute_local[3] + A_shared_local[3]);
    }
  }
  compute[((((int)blockIdx.x) * 512) + ((int)threadIdx.x))] = compute_local[0];
  compute[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 128)] = compute_local[1];
  compute[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 256)] = compute_local[2];
  compute[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 384)] = compute_local[3];
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
    int block_size = 128;
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
