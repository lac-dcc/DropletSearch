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
extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ A, float* __restrict__ compute) {
  float compute_local[2];
  __shared__ float A_shared[4096];
  float A_shared_local[2];
  compute_local[0] = 0.000000e+00f;
  compute_local[1] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    __syncthreads();
    A_shared[((int)threadIdx.x)] = A[(((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x))];
    A_shared[(((int)threadIdx.x) + 32)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 32)];
    A_shared[(((int)threadIdx.x) + 64)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 1024)];
    A_shared[(((int)threadIdx.x) + 96)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 1056)];
    A_shared[(((int)threadIdx.x) + 128)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 2048)];
    A_shared[(((int)threadIdx.x) + 160)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 2080)];
    A_shared[(((int)threadIdx.x) + 192)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 3072)];
    A_shared[(((int)threadIdx.x) + 224)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 3104)];
    A_shared[(((int)threadIdx.x) + 256)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 4096)];
    A_shared[(((int)threadIdx.x) + 288)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 4128)];
    A_shared[(((int)threadIdx.x) + 320)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 5120)];
    A_shared[(((int)threadIdx.x) + 352)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 5152)];
    A_shared[(((int)threadIdx.x) + 384)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 6144)];
    A_shared[(((int)threadIdx.x) + 416)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 6176)];
    A_shared[(((int)threadIdx.x) + 448)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 7168)];
    A_shared[(((int)threadIdx.x) + 480)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 7200)];
    A_shared[(((int)threadIdx.x) + 512)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 8192)];
    A_shared[(((int)threadIdx.x) + 544)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 8224)];
    A_shared[(((int)threadIdx.x) + 576)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 9216)];
    A_shared[(((int)threadIdx.x) + 608)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 9248)];
    A_shared[(((int)threadIdx.x) + 640)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 10240)];
    A_shared[(((int)threadIdx.x) + 672)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 10272)];
    A_shared[(((int)threadIdx.x) + 704)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 11264)];
    A_shared[(((int)threadIdx.x) + 736)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 11296)];
    A_shared[(((int)threadIdx.x) + 768)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 12288)];
    A_shared[(((int)threadIdx.x) + 800)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 12320)];
    A_shared[(((int)threadIdx.x) + 832)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 13312)];
    A_shared[(((int)threadIdx.x) + 864)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 13344)];
    A_shared[(((int)threadIdx.x) + 896)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 14336)];
    A_shared[(((int)threadIdx.x) + 928)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 14368)];
    A_shared[(((int)threadIdx.x) + 960)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 15360)];
    A_shared[(((int)threadIdx.x) + 992)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 15392)];
    A_shared[(((int)threadIdx.x) + 1024)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 16384)];
    A_shared[(((int)threadIdx.x) + 1056)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 16416)];
    A_shared[(((int)threadIdx.x) + 1088)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 17408)];
    A_shared[(((int)threadIdx.x) + 1120)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 17440)];
    A_shared[(((int)threadIdx.x) + 1152)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 18432)];
    A_shared[(((int)threadIdx.x) + 1184)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 18464)];
    A_shared[(((int)threadIdx.x) + 1216)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 19456)];
    A_shared[(((int)threadIdx.x) + 1248)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 19488)];
    A_shared[(((int)threadIdx.x) + 1280)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 20480)];
    A_shared[(((int)threadIdx.x) + 1312)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 20512)];
    A_shared[(((int)threadIdx.x) + 1344)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 21504)];
    A_shared[(((int)threadIdx.x) + 1376)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 21536)];
    A_shared[(((int)threadIdx.x) + 1408)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 22528)];
    A_shared[(((int)threadIdx.x) + 1440)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 22560)];
    A_shared[(((int)threadIdx.x) + 1472)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 23552)];
    A_shared[(((int)threadIdx.x) + 1504)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 23584)];
    A_shared[(((int)threadIdx.x) + 1536)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 24576)];
    A_shared[(((int)threadIdx.x) + 1568)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 24608)];
    A_shared[(((int)threadIdx.x) + 1600)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 25600)];
    A_shared[(((int)threadIdx.x) + 1632)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 25632)];
    A_shared[(((int)threadIdx.x) + 1664)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 26624)];
    A_shared[(((int)threadIdx.x) + 1696)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 26656)];
    A_shared[(((int)threadIdx.x) + 1728)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 27648)];
    A_shared[(((int)threadIdx.x) + 1760)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 27680)];
    A_shared[(((int)threadIdx.x) + 1792)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 28672)];
    A_shared[(((int)threadIdx.x) + 1824)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 28704)];
    A_shared[(((int)threadIdx.x) + 1856)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 29696)];
    A_shared[(((int)threadIdx.x) + 1888)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 29728)];
    A_shared[(((int)threadIdx.x) + 1920)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 30720)];
    A_shared[(((int)threadIdx.x) + 1952)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 30752)];
    A_shared[(((int)threadIdx.x) + 1984)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 31744)];
    A_shared[(((int)threadIdx.x) + 2016)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 31776)];
    A_shared[(((int)threadIdx.x) + 2048)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 32768)];
    A_shared[(((int)threadIdx.x) + 2080)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 32800)];
    A_shared[(((int)threadIdx.x) + 2112)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 33792)];
    A_shared[(((int)threadIdx.x) + 2144)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 33824)];
    A_shared[(((int)threadIdx.x) + 2176)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 34816)];
    A_shared[(((int)threadIdx.x) + 2208)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 34848)];
    A_shared[(((int)threadIdx.x) + 2240)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 35840)];
    A_shared[(((int)threadIdx.x) + 2272)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 35872)];
    A_shared[(((int)threadIdx.x) + 2304)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 36864)];
    A_shared[(((int)threadIdx.x) + 2336)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 36896)];
    A_shared[(((int)threadIdx.x) + 2368)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 37888)];
    A_shared[(((int)threadIdx.x) + 2400)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 37920)];
    A_shared[(((int)threadIdx.x) + 2432)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 38912)];
    A_shared[(((int)threadIdx.x) + 2464)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 38944)];
    A_shared[(((int)threadIdx.x) + 2496)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 39936)];
    A_shared[(((int)threadIdx.x) + 2528)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 39968)];
    A_shared[(((int)threadIdx.x) + 2560)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 40960)];
    A_shared[(((int)threadIdx.x) + 2592)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 40992)];
    A_shared[(((int)threadIdx.x) + 2624)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 41984)];
    A_shared[(((int)threadIdx.x) + 2656)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 42016)];
    A_shared[(((int)threadIdx.x) + 2688)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 43008)];
    A_shared[(((int)threadIdx.x) + 2720)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 43040)];
    A_shared[(((int)threadIdx.x) + 2752)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 44032)];
    A_shared[(((int)threadIdx.x) + 2784)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 44064)];
    A_shared[(((int)threadIdx.x) + 2816)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 45056)];
    A_shared[(((int)threadIdx.x) + 2848)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 45088)];
    A_shared[(((int)threadIdx.x) + 2880)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 46080)];
    A_shared[(((int)threadIdx.x) + 2912)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 46112)];
    A_shared[(((int)threadIdx.x) + 2944)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 47104)];
    A_shared[(((int)threadIdx.x) + 2976)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 47136)];
    A_shared[(((int)threadIdx.x) + 3008)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 48128)];
    A_shared[(((int)threadIdx.x) + 3040)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 48160)];
    A_shared[(((int)threadIdx.x) + 3072)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 49152)];
    A_shared[(((int)threadIdx.x) + 3104)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 49184)];
    A_shared[(((int)threadIdx.x) + 3136)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 50176)];
    A_shared[(((int)threadIdx.x) + 3168)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 50208)];
    A_shared[(((int)threadIdx.x) + 3200)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 51200)];
    A_shared[(((int)threadIdx.x) + 3232)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 51232)];
    A_shared[(((int)threadIdx.x) + 3264)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 52224)];
    A_shared[(((int)threadIdx.x) + 3296)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 52256)];
    A_shared[(((int)threadIdx.x) + 3328)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 53248)];
    A_shared[(((int)threadIdx.x) + 3360)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 53280)];
    A_shared[(((int)threadIdx.x) + 3392)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 54272)];
    A_shared[(((int)threadIdx.x) + 3424)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 54304)];
    A_shared[(((int)threadIdx.x) + 3456)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 55296)];
    A_shared[(((int)threadIdx.x) + 3488)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 55328)];
    A_shared[(((int)threadIdx.x) + 3520)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 56320)];
    A_shared[(((int)threadIdx.x) + 3552)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 56352)];
    A_shared[(((int)threadIdx.x) + 3584)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 57344)];
    A_shared[(((int)threadIdx.x) + 3616)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 57376)];
    A_shared[(((int)threadIdx.x) + 3648)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 58368)];
    A_shared[(((int)threadIdx.x) + 3680)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 58400)];
    A_shared[(((int)threadIdx.x) + 3712)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 59392)];
    A_shared[(((int)threadIdx.x) + 3744)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 59424)];
    A_shared[(((int)threadIdx.x) + 3776)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 60416)];
    A_shared[(((int)threadIdx.x) + 3808)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 60448)];
    A_shared[(((int)threadIdx.x) + 3840)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 61440)];
    A_shared[(((int)threadIdx.x) + 3872)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 61472)];
    A_shared[(((int)threadIdx.x) + 3904)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 62464)];
    A_shared[(((int)threadIdx.x) + 3936)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 62496)];
    A_shared[(((int)threadIdx.x) + 3968)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 63488)];
    A_shared[(((int)threadIdx.x) + 4000)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 63520)];
    A_shared[(((int)threadIdx.x) + 4032)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 64512)];
    A_shared[(((int)threadIdx.x) + 4064)] = A[((((((int)blockIdx.x) * 65536) + (k_outer * 64)) + ((int)threadIdx.x)) + 64544)];
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 64; ++k_inner_outer) {
      A_shared_local[0] = A_shared[((((int)threadIdx.x) * 64) + k_inner_outer)];
      A_shared_local[1] = A_shared[(((((int)threadIdx.x) * 64) + k_inner_outer) + 2048)];
      compute_local[0] = (compute_local[0] + A_shared_local[0]);
      compute_local[1] = (compute_local[1] + A_shared_local[1]);
    }
  }
  compute[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = compute_local[0];
  compute[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) + 32)] = compute_local[1];
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

    int grid_size = 1024;
    int block_size = 32;
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
