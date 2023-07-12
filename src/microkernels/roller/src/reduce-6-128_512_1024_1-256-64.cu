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
extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ A, float* __restrict__ compute) {
  float compute_local[4];
  __shared__ float A_shared[8192];
  float A_shared_local[4];
  compute_local[0] = 0.000000e+00f;
  compute_local[1] = 0.000000e+00f;
  compute_local[2] = 0.000000e+00f;
  compute_local[3] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 32; ++k_outer) {
    __syncthreads();
    A_shared[((int)threadIdx.x)] = A[((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31))];
    A_shared[(((int)threadIdx.x) + 64)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 2048)];
    A_shared[(((int)threadIdx.x) + 128)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 4096)];
    A_shared[(((int)threadIdx.x) + 192)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 6144)];
    A_shared[(((int)threadIdx.x) + 256)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 8192)];
    A_shared[(((int)threadIdx.x) + 320)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 10240)];
    A_shared[(((int)threadIdx.x) + 384)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 12288)];
    A_shared[(((int)threadIdx.x) + 448)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 14336)];
    A_shared[(((int)threadIdx.x) + 512)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 16384)];
    A_shared[(((int)threadIdx.x) + 576)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 18432)];
    A_shared[(((int)threadIdx.x) + 640)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 20480)];
    A_shared[(((int)threadIdx.x) + 704)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 22528)];
    A_shared[(((int)threadIdx.x) + 768)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 24576)];
    A_shared[(((int)threadIdx.x) + 832)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 26624)];
    A_shared[(((int)threadIdx.x) + 896)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 28672)];
    A_shared[(((int)threadIdx.x) + 960)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 30720)];
    A_shared[(((int)threadIdx.x) + 1024)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 32768)];
    A_shared[(((int)threadIdx.x) + 1088)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 34816)];
    A_shared[(((int)threadIdx.x) + 1152)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 36864)];
    A_shared[(((int)threadIdx.x) + 1216)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 38912)];
    A_shared[(((int)threadIdx.x) + 1280)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 40960)];
    A_shared[(((int)threadIdx.x) + 1344)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 43008)];
    A_shared[(((int)threadIdx.x) + 1408)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 45056)];
    A_shared[(((int)threadIdx.x) + 1472)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 47104)];
    A_shared[(((int)threadIdx.x) + 1536)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 49152)];
    A_shared[(((int)threadIdx.x) + 1600)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 51200)];
    A_shared[(((int)threadIdx.x) + 1664)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 53248)];
    A_shared[(((int)threadIdx.x) + 1728)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 55296)];
    A_shared[(((int)threadIdx.x) + 1792)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 57344)];
    A_shared[(((int)threadIdx.x) + 1856)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 59392)];
    A_shared[(((int)threadIdx.x) + 1920)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 61440)];
    A_shared[(((int)threadIdx.x) + 1984)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 63488)];
    A_shared[(((int)threadIdx.x) + 2048)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 65536)];
    A_shared[(((int)threadIdx.x) + 2112)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 67584)];
    A_shared[(((int)threadIdx.x) + 2176)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 69632)];
    A_shared[(((int)threadIdx.x) + 2240)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 71680)];
    A_shared[(((int)threadIdx.x) + 2304)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 73728)];
    A_shared[(((int)threadIdx.x) + 2368)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 75776)];
    A_shared[(((int)threadIdx.x) + 2432)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 77824)];
    A_shared[(((int)threadIdx.x) + 2496)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 79872)];
    A_shared[(((int)threadIdx.x) + 2560)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 81920)];
    A_shared[(((int)threadIdx.x) + 2624)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 83968)];
    A_shared[(((int)threadIdx.x) + 2688)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 86016)];
    A_shared[(((int)threadIdx.x) + 2752)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 88064)];
    A_shared[(((int)threadIdx.x) + 2816)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 90112)];
    A_shared[(((int)threadIdx.x) + 2880)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 92160)];
    A_shared[(((int)threadIdx.x) + 2944)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 94208)];
    A_shared[(((int)threadIdx.x) + 3008)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 96256)];
    A_shared[(((int)threadIdx.x) + 3072)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 98304)];
    A_shared[(((int)threadIdx.x) + 3136)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 100352)];
    A_shared[(((int)threadIdx.x) + 3200)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 102400)];
    A_shared[(((int)threadIdx.x) + 3264)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 104448)];
    A_shared[(((int)threadIdx.x) + 3328)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 106496)];
    A_shared[(((int)threadIdx.x) + 3392)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 108544)];
    A_shared[(((int)threadIdx.x) + 3456)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 110592)];
    A_shared[(((int)threadIdx.x) + 3520)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 112640)];
    A_shared[(((int)threadIdx.x) + 3584)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 114688)];
    A_shared[(((int)threadIdx.x) + 3648)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 116736)];
    A_shared[(((int)threadIdx.x) + 3712)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 118784)];
    A_shared[(((int)threadIdx.x) + 3776)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 120832)];
    A_shared[(((int)threadIdx.x) + 3840)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 122880)];
    A_shared[(((int)threadIdx.x) + 3904)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 124928)];
    A_shared[(((int)threadIdx.x) + 3968)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 126976)];
    A_shared[(((int)threadIdx.x) + 4032)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 129024)];
    A_shared[(((int)threadIdx.x) + 4096)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 131072)];
    A_shared[(((int)threadIdx.x) + 4160)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 133120)];
    A_shared[(((int)threadIdx.x) + 4224)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 135168)];
    A_shared[(((int)threadIdx.x) + 4288)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 137216)];
    A_shared[(((int)threadIdx.x) + 4352)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 139264)];
    A_shared[(((int)threadIdx.x) + 4416)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 141312)];
    A_shared[(((int)threadIdx.x) + 4480)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 143360)];
    A_shared[(((int)threadIdx.x) + 4544)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 145408)];
    A_shared[(((int)threadIdx.x) + 4608)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 147456)];
    A_shared[(((int)threadIdx.x) + 4672)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 149504)];
    A_shared[(((int)threadIdx.x) + 4736)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 151552)];
    A_shared[(((int)threadIdx.x) + 4800)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 153600)];
    A_shared[(((int)threadIdx.x) + 4864)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 155648)];
    A_shared[(((int)threadIdx.x) + 4928)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 157696)];
    A_shared[(((int)threadIdx.x) + 4992)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 159744)];
    A_shared[(((int)threadIdx.x) + 5056)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 161792)];
    A_shared[(((int)threadIdx.x) + 5120)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 163840)];
    A_shared[(((int)threadIdx.x) + 5184)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 165888)];
    A_shared[(((int)threadIdx.x) + 5248)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 167936)];
    A_shared[(((int)threadIdx.x) + 5312)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 169984)];
    A_shared[(((int)threadIdx.x) + 5376)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 172032)];
    A_shared[(((int)threadIdx.x) + 5440)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 174080)];
    A_shared[(((int)threadIdx.x) + 5504)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 176128)];
    A_shared[(((int)threadIdx.x) + 5568)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 178176)];
    A_shared[(((int)threadIdx.x) + 5632)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 180224)];
    A_shared[(((int)threadIdx.x) + 5696)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 182272)];
    A_shared[(((int)threadIdx.x) + 5760)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 184320)];
    A_shared[(((int)threadIdx.x) + 5824)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 186368)];
    A_shared[(((int)threadIdx.x) + 5888)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 188416)];
    A_shared[(((int)threadIdx.x) + 5952)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 190464)];
    A_shared[(((int)threadIdx.x) + 6016)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 192512)];
    A_shared[(((int)threadIdx.x) + 6080)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 194560)];
    A_shared[(((int)threadIdx.x) + 6144)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 196608)];
    A_shared[(((int)threadIdx.x) + 6208)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 198656)];
    A_shared[(((int)threadIdx.x) + 6272)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 200704)];
    A_shared[(((int)threadIdx.x) + 6336)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 202752)];
    A_shared[(((int)threadIdx.x) + 6400)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 204800)];
    A_shared[(((int)threadIdx.x) + 6464)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 206848)];
    A_shared[(((int)threadIdx.x) + 6528)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 208896)];
    A_shared[(((int)threadIdx.x) + 6592)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 210944)];
    A_shared[(((int)threadIdx.x) + 6656)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 212992)];
    A_shared[(((int)threadIdx.x) + 6720)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 215040)];
    A_shared[(((int)threadIdx.x) + 6784)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 217088)];
    A_shared[(((int)threadIdx.x) + 6848)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 219136)];
    A_shared[(((int)threadIdx.x) + 6912)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 221184)];
    A_shared[(((int)threadIdx.x) + 6976)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 223232)];
    A_shared[(((int)threadIdx.x) + 7040)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 225280)];
    A_shared[(((int)threadIdx.x) + 7104)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 227328)];
    A_shared[(((int)threadIdx.x) + 7168)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 229376)];
    A_shared[(((int)threadIdx.x) + 7232)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 231424)];
    A_shared[(((int)threadIdx.x) + 7296)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 233472)];
    A_shared[(((int)threadIdx.x) + 7360)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 235520)];
    A_shared[(((int)threadIdx.x) + 7424)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 237568)];
    A_shared[(((int)threadIdx.x) + 7488)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 239616)];
    A_shared[(((int)threadIdx.x) + 7552)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 241664)];
    A_shared[(((int)threadIdx.x) + 7616)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 243712)];
    A_shared[(((int)threadIdx.x) + 7680)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 245760)];
    A_shared[(((int)threadIdx.x) + 7744)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 247808)];
    A_shared[(((int)threadIdx.x) + 7808)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 249856)];
    A_shared[(((int)threadIdx.x) + 7872)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 251904)];
    A_shared[(((int)threadIdx.x) + 7936)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 253952)];
    A_shared[(((int)threadIdx.x) + 8000)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 256000)];
    A_shared[(((int)threadIdx.x) + 8064)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 258048)];
    A_shared[(((int)threadIdx.x) + 8128)] = A[(((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 260096)];
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 32; ++k_inner_outer) {
      A_shared_local[0] = A_shared[((((int)threadIdx.x) * 32) + k_inner_outer)];
      A_shared_local[1] = A_shared[(((((int)threadIdx.x) * 32) + k_inner_outer) + 2048)];
      A_shared_local[2] = A_shared[(((((int)threadIdx.x) * 32) + k_inner_outer) + 4096)];
      A_shared_local[3] = A_shared[(((((int)threadIdx.x) * 32) + k_inner_outer) + 6144)];
      compute_local[0] = (compute_local[0] + A_shared_local[0]);
      compute_local[1] = (compute_local[1] + A_shared_local[1]);
      compute_local[2] = (compute_local[2] + A_shared_local[2]);
      compute_local[3] = (compute_local[3] + A_shared_local[3]);
    }
  }
  compute[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] = compute_local[0];
  compute[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 64)] = compute_local[1];
  compute[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 128)] = compute_local[2];
  compute[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 192)] = compute_local[3];
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

    int grid_size = 256;
    int block_size = 64;
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
