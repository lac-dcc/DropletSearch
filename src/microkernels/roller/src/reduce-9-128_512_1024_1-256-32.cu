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
  float compute_local[8];
  __shared__ float A_shared[8192];
  float A_shared_local[8];
  compute_local[0] = 0.000000e+00f;
  compute_local[1] = 0.000000e+00f;
  compute_local[2] = 0.000000e+00f;
  compute_local[3] = 0.000000e+00f;
  compute_local[4] = 0.000000e+00f;
  compute_local[5] = 0.000000e+00f;
  compute_local[6] = 0.000000e+00f;
  compute_local[7] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 32; ++k_outer) {
    __syncthreads();
    A_shared[((int)threadIdx.x)] = A[(((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x))];
    A_shared[(((int)threadIdx.x) + 32)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 1024)];
    A_shared[(((int)threadIdx.x) + 64)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 2048)];
    A_shared[(((int)threadIdx.x) + 96)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 3072)];
    A_shared[(((int)threadIdx.x) + 128)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 4096)];
    A_shared[(((int)threadIdx.x) + 160)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 5120)];
    A_shared[(((int)threadIdx.x) + 192)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 6144)];
    A_shared[(((int)threadIdx.x) + 224)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 7168)];
    A_shared[(((int)threadIdx.x) + 256)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 8192)];
    A_shared[(((int)threadIdx.x) + 288)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 9216)];
    A_shared[(((int)threadIdx.x) + 320)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 10240)];
    A_shared[(((int)threadIdx.x) + 352)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 11264)];
    A_shared[(((int)threadIdx.x) + 384)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 12288)];
    A_shared[(((int)threadIdx.x) + 416)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 13312)];
    A_shared[(((int)threadIdx.x) + 448)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 14336)];
    A_shared[(((int)threadIdx.x) + 480)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 15360)];
    A_shared[(((int)threadIdx.x) + 512)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 16384)];
    A_shared[(((int)threadIdx.x) + 544)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 17408)];
    A_shared[(((int)threadIdx.x) + 576)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 18432)];
    A_shared[(((int)threadIdx.x) + 608)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 19456)];
    A_shared[(((int)threadIdx.x) + 640)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 20480)];
    A_shared[(((int)threadIdx.x) + 672)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 21504)];
    A_shared[(((int)threadIdx.x) + 704)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 22528)];
    A_shared[(((int)threadIdx.x) + 736)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 23552)];
    A_shared[(((int)threadIdx.x) + 768)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 24576)];
    A_shared[(((int)threadIdx.x) + 800)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 25600)];
    A_shared[(((int)threadIdx.x) + 832)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 26624)];
    A_shared[(((int)threadIdx.x) + 864)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 27648)];
    A_shared[(((int)threadIdx.x) + 896)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 28672)];
    A_shared[(((int)threadIdx.x) + 928)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 29696)];
    A_shared[(((int)threadIdx.x) + 960)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 30720)];
    A_shared[(((int)threadIdx.x) + 992)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 31744)];
    A_shared[(((int)threadIdx.x) + 1024)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 32768)];
    A_shared[(((int)threadIdx.x) + 1056)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 33792)];
    A_shared[(((int)threadIdx.x) + 1088)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 34816)];
    A_shared[(((int)threadIdx.x) + 1120)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 35840)];
    A_shared[(((int)threadIdx.x) + 1152)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 36864)];
    A_shared[(((int)threadIdx.x) + 1184)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 37888)];
    A_shared[(((int)threadIdx.x) + 1216)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 38912)];
    A_shared[(((int)threadIdx.x) + 1248)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 39936)];
    A_shared[(((int)threadIdx.x) + 1280)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 40960)];
    A_shared[(((int)threadIdx.x) + 1312)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 41984)];
    A_shared[(((int)threadIdx.x) + 1344)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 43008)];
    A_shared[(((int)threadIdx.x) + 1376)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 44032)];
    A_shared[(((int)threadIdx.x) + 1408)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 45056)];
    A_shared[(((int)threadIdx.x) + 1440)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 46080)];
    A_shared[(((int)threadIdx.x) + 1472)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 47104)];
    A_shared[(((int)threadIdx.x) + 1504)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 48128)];
    A_shared[(((int)threadIdx.x) + 1536)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 49152)];
    A_shared[(((int)threadIdx.x) + 1568)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 50176)];
    A_shared[(((int)threadIdx.x) + 1600)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 51200)];
    A_shared[(((int)threadIdx.x) + 1632)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 52224)];
    A_shared[(((int)threadIdx.x) + 1664)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 53248)];
    A_shared[(((int)threadIdx.x) + 1696)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 54272)];
    A_shared[(((int)threadIdx.x) + 1728)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 55296)];
    A_shared[(((int)threadIdx.x) + 1760)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 56320)];
    A_shared[(((int)threadIdx.x) + 1792)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 57344)];
    A_shared[(((int)threadIdx.x) + 1824)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 58368)];
    A_shared[(((int)threadIdx.x) + 1856)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 59392)];
    A_shared[(((int)threadIdx.x) + 1888)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 60416)];
    A_shared[(((int)threadIdx.x) + 1920)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 61440)];
    A_shared[(((int)threadIdx.x) + 1952)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 62464)];
    A_shared[(((int)threadIdx.x) + 1984)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 63488)];
    A_shared[(((int)threadIdx.x) + 2016)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 64512)];
    A_shared[(((int)threadIdx.x) + 2048)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 65536)];
    A_shared[(((int)threadIdx.x) + 2080)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 66560)];
    A_shared[(((int)threadIdx.x) + 2112)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 67584)];
    A_shared[(((int)threadIdx.x) + 2144)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 68608)];
    A_shared[(((int)threadIdx.x) + 2176)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 69632)];
    A_shared[(((int)threadIdx.x) + 2208)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 70656)];
    A_shared[(((int)threadIdx.x) + 2240)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 71680)];
    A_shared[(((int)threadIdx.x) + 2272)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 72704)];
    A_shared[(((int)threadIdx.x) + 2304)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 73728)];
    A_shared[(((int)threadIdx.x) + 2336)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 74752)];
    A_shared[(((int)threadIdx.x) + 2368)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 75776)];
    A_shared[(((int)threadIdx.x) + 2400)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 76800)];
    A_shared[(((int)threadIdx.x) + 2432)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 77824)];
    A_shared[(((int)threadIdx.x) + 2464)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 78848)];
    A_shared[(((int)threadIdx.x) + 2496)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 79872)];
    A_shared[(((int)threadIdx.x) + 2528)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 80896)];
    A_shared[(((int)threadIdx.x) + 2560)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 81920)];
    A_shared[(((int)threadIdx.x) + 2592)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 82944)];
    A_shared[(((int)threadIdx.x) + 2624)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 83968)];
    A_shared[(((int)threadIdx.x) + 2656)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 84992)];
    A_shared[(((int)threadIdx.x) + 2688)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 86016)];
    A_shared[(((int)threadIdx.x) + 2720)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 87040)];
    A_shared[(((int)threadIdx.x) + 2752)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 88064)];
    A_shared[(((int)threadIdx.x) + 2784)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 89088)];
    A_shared[(((int)threadIdx.x) + 2816)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 90112)];
    A_shared[(((int)threadIdx.x) + 2848)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 91136)];
    A_shared[(((int)threadIdx.x) + 2880)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 92160)];
    A_shared[(((int)threadIdx.x) + 2912)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 93184)];
    A_shared[(((int)threadIdx.x) + 2944)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 94208)];
    A_shared[(((int)threadIdx.x) + 2976)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 95232)];
    A_shared[(((int)threadIdx.x) + 3008)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 96256)];
    A_shared[(((int)threadIdx.x) + 3040)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 97280)];
    A_shared[(((int)threadIdx.x) + 3072)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 98304)];
    A_shared[(((int)threadIdx.x) + 3104)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 99328)];
    A_shared[(((int)threadIdx.x) + 3136)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 100352)];
    A_shared[(((int)threadIdx.x) + 3168)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 101376)];
    A_shared[(((int)threadIdx.x) + 3200)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 102400)];
    A_shared[(((int)threadIdx.x) + 3232)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 103424)];
    A_shared[(((int)threadIdx.x) + 3264)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 104448)];
    A_shared[(((int)threadIdx.x) + 3296)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 105472)];
    A_shared[(((int)threadIdx.x) + 3328)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 106496)];
    A_shared[(((int)threadIdx.x) + 3360)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 107520)];
    A_shared[(((int)threadIdx.x) + 3392)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 108544)];
    A_shared[(((int)threadIdx.x) + 3424)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 109568)];
    A_shared[(((int)threadIdx.x) + 3456)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 110592)];
    A_shared[(((int)threadIdx.x) + 3488)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 111616)];
    A_shared[(((int)threadIdx.x) + 3520)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 112640)];
    A_shared[(((int)threadIdx.x) + 3552)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 113664)];
    A_shared[(((int)threadIdx.x) + 3584)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 114688)];
    A_shared[(((int)threadIdx.x) + 3616)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 115712)];
    A_shared[(((int)threadIdx.x) + 3648)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 116736)];
    A_shared[(((int)threadIdx.x) + 3680)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 117760)];
    A_shared[(((int)threadIdx.x) + 3712)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 118784)];
    A_shared[(((int)threadIdx.x) + 3744)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 119808)];
    A_shared[(((int)threadIdx.x) + 3776)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 120832)];
    A_shared[(((int)threadIdx.x) + 3808)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 121856)];
    A_shared[(((int)threadIdx.x) + 3840)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 122880)];
    A_shared[(((int)threadIdx.x) + 3872)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 123904)];
    A_shared[(((int)threadIdx.x) + 3904)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 124928)];
    A_shared[(((int)threadIdx.x) + 3936)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 125952)];
    A_shared[(((int)threadIdx.x) + 3968)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 126976)];
    A_shared[(((int)threadIdx.x) + 4000)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 128000)];
    A_shared[(((int)threadIdx.x) + 4032)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 129024)];
    A_shared[(((int)threadIdx.x) + 4064)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 130048)];
    A_shared[(((int)threadIdx.x) + 4096)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 131072)];
    A_shared[(((int)threadIdx.x) + 4128)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 132096)];
    A_shared[(((int)threadIdx.x) + 4160)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 133120)];
    A_shared[(((int)threadIdx.x) + 4192)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 134144)];
    A_shared[(((int)threadIdx.x) + 4224)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 135168)];
    A_shared[(((int)threadIdx.x) + 4256)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 136192)];
    A_shared[(((int)threadIdx.x) + 4288)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 137216)];
    A_shared[(((int)threadIdx.x) + 4320)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 138240)];
    A_shared[(((int)threadIdx.x) + 4352)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 139264)];
    A_shared[(((int)threadIdx.x) + 4384)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 140288)];
    A_shared[(((int)threadIdx.x) + 4416)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 141312)];
    A_shared[(((int)threadIdx.x) + 4448)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 142336)];
    A_shared[(((int)threadIdx.x) + 4480)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 143360)];
    A_shared[(((int)threadIdx.x) + 4512)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 144384)];
    A_shared[(((int)threadIdx.x) + 4544)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 145408)];
    A_shared[(((int)threadIdx.x) + 4576)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 146432)];
    A_shared[(((int)threadIdx.x) + 4608)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 147456)];
    A_shared[(((int)threadIdx.x) + 4640)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 148480)];
    A_shared[(((int)threadIdx.x) + 4672)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 149504)];
    A_shared[(((int)threadIdx.x) + 4704)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 150528)];
    A_shared[(((int)threadIdx.x) + 4736)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 151552)];
    A_shared[(((int)threadIdx.x) + 4768)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 152576)];
    A_shared[(((int)threadIdx.x) + 4800)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 153600)];
    A_shared[(((int)threadIdx.x) + 4832)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 154624)];
    A_shared[(((int)threadIdx.x) + 4864)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 155648)];
    A_shared[(((int)threadIdx.x) + 4896)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 156672)];
    A_shared[(((int)threadIdx.x) + 4928)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 157696)];
    A_shared[(((int)threadIdx.x) + 4960)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 158720)];
    A_shared[(((int)threadIdx.x) + 4992)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 159744)];
    A_shared[(((int)threadIdx.x) + 5024)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 160768)];
    A_shared[(((int)threadIdx.x) + 5056)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 161792)];
    A_shared[(((int)threadIdx.x) + 5088)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 162816)];
    A_shared[(((int)threadIdx.x) + 5120)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 163840)];
    A_shared[(((int)threadIdx.x) + 5152)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 164864)];
    A_shared[(((int)threadIdx.x) + 5184)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 165888)];
    A_shared[(((int)threadIdx.x) + 5216)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 166912)];
    A_shared[(((int)threadIdx.x) + 5248)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 167936)];
    A_shared[(((int)threadIdx.x) + 5280)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 168960)];
    A_shared[(((int)threadIdx.x) + 5312)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 169984)];
    A_shared[(((int)threadIdx.x) + 5344)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 171008)];
    A_shared[(((int)threadIdx.x) + 5376)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 172032)];
    A_shared[(((int)threadIdx.x) + 5408)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 173056)];
    A_shared[(((int)threadIdx.x) + 5440)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 174080)];
    A_shared[(((int)threadIdx.x) + 5472)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 175104)];
    A_shared[(((int)threadIdx.x) + 5504)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 176128)];
    A_shared[(((int)threadIdx.x) + 5536)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 177152)];
    A_shared[(((int)threadIdx.x) + 5568)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 178176)];
    A_shared[(((int)threadIdx.x) + 5600)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 179200)];
    A_shared[(((int)threadIdx.x) + 5632)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 180224)];
    A_shared[(((int)threadIdx.x) + 5664)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 181248)];
    A_shared[(((int)threadIdx.x) + 5696)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 182272)];
    A_shared[(((int)threadIdx.x) + 5728)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 183296)];
    A_shared[(((int)threadIdx.x) + 5760)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 184320)];
    A_shared[(((int)threadIdx.x) + 5792)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 185344)];
    A_shared[(((int)threadIdx.x) + 5824)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 186368)];
    A_shared[(((int)threadIdx.x) + 5856)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 187392)];
    A_shared[(((int)threadIdx.x) + 5888)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 188416)];
    A_shared[(((int)threadIdx.x) + 5920)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 189440)];
    A_shared[(((int)threadIdx.x) + 5952)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 190464)];
    A_shared[(((int)threadIdx.x) + 5984)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 191488)];
    A_shared[(((int)threadIdx.x) + 6016)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 192512)];
    A_shared[(((int)threadIdx.x) + 6048)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 193536)];
    A_shared[(((int)threadIdx.x) + 6080)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 194560)];
    A_shared[(((int)threadIdx.x) + 6112)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 195584)];
    A_shared[(((int)threadIdx.x) + 6144)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 196608)];
    A_shared[(((int)threadIdx.x) + 6176)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 197632)];
    A_shared[(((int)threadIdx.x) + 6208)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 198656)];
    A_shared[(((int)threadIdx.x) + 6240)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 199680)];
    A_shared[(((int)threadIdx.x) + 6272)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 200704)];
    A_shared[(((int)threadIdx.x) + 6304)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 201728)];
    A_shared[(((int)threadIdx.x) + 6336)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 202752)];
    A_shared[(((int)threadIdx.x) + 6368)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 203776)];
    A_shared[(((int)threadIdx.x) + 6400)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 204800)];
    A_shared[(((int)threadIdx.x) + 6432)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 205824)];
    A_shared[(((int)threadIdx.x) + 6464)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 206848)];
    A_shared[(((int)threadIdx.x) + 6496)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 207872)];
    A_shared[(((int)threadIdx.x) + 6528)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 208896)];
    A_shared[(((int)threadIdx.x) + 6560)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 209920)];
    A_shared[(((int)threadIdx.x) + 6592)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 210944)];
    A_shared[(((int)threadIdx.x) + 6624)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 211968)];
    A_shared[(((int)threadIdx.x) + 6656)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 212992)];
    A_shared[(((int)threadIdx.x) + 6688)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 214016)];
    A_shared[(((int)threadIdx.x) + 6720)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 215040)];
    A_shared[(((int)threadIdx.x) + 6752)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 216064)];
    A_shared[(((int)threadIdx.x) + 6784)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 217088)];
    A_shared[(((int)threadIdx.x) + 6816)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 218112)];
    A_shared[(((int)threadIdx.x) + 6848)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 219136)];
    A_shared[(((int)threadIdx.x) + 6880)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 220160)];
    A_shared[(((int)threadIdx.x) + 6912)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 221184)];
    A_shared[(((int)threadIdx.x) + 6944)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 222208)];
    A_shared[(((int)threadIdx.x) + 6976)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 223232)];
    A_shared[(((int)threadIdx.x) + 7008)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 224256)];
    A_shared[(((int)threadIdx.x) + 7040)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 225280)];
    A_shared[(((int)threadIdx.x) + 7072)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 226304)];
    A_shared[(((int)threadIdx.x) + 7104)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 227328)];
    A_shared[(((int)threadIdx.x) + 7136)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 228352)];
    A_shared[(((int)threadIdx.x) + 7168)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 229376)];
    A_shared[(((int)threadIdx.x) + 7200)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 230400)];
    A_shared[(((int)threadIdx.x) + 7232)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 231424)];
    A_shared[(((int)threadIdx.x) + 7264)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 232448)];
    A_shared[(((int)threadIdx.x) + 7296)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 233472)];
    A_shared[(((int)threadIdx.x) + 7328)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 234496)];
    A_shared[(((int)threadIdx.x) + 7360)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 235520)];
    A_shared[(((int)threadIdx.x) + 7392)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 236544)];
    A_shared[(((int)threadIdx.x) + 7424)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 237568)];
    A_shared[(((int)threadIdx.x) + 7456)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 238592)];
    A_shared[(((int)threadIdx.x) + 7488)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 239616)];
    A_shared[(((int)threadIdx.x) + 7520)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 240640)];
    A_shared[(((int)threadIdx.x) + 7552)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 241664)];
    A_shared[(((int)threadIdx.x) + 7584)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 242688)];
    A_shared[(((int)threadIdx.x) + 7616)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 243712)];
    A_shared[(((int)threadIdx.x) + 7648)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 244736)];
    A_shared[(((int)threadIdx.x) + 7680)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 245760)];
    A_shared[(((int)threadIdx.x) + 7712)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 246784)];
    A_shared[(((int)threadIdx.x) + 7744)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 247808)];
    A_shared[(((int)threadIdx.x) + 7776)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 248832)];
    A_shared[(((int)threadIdx.x) + 7808)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 249856)];
    A_shared[(((int)threadIdx.x) + 7840)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 250880)];
    A_shared[(((int)threadIdx.x) + 7872)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 251904)];
    A_shared[(((int)threadIdx.x) + 7904)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 252928)];
    A_shared[(((int)threadIdx.x) + 7936)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 253952)];
    A_shared[(((int)threadIdx.x) + 7968)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 254976)];
    A_shared[(((int)threadIdx.x) + 8000)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 256000)];
    A_shared[(((int)threadIdx.x) + 8032)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 257024)];
    A_shared[(((int)threadIdx.x) + 8064)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 258048)];
    A_shared[(((int)threadIdx.x) + 8096)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 259072)];
    A_shared[(((int)threadIdx.x) + 8128)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 260096)];
    A_shared[(((int)threadIdx.x) + 8160)] = A[((((((int)blockIdx.x) * 262144) + (k_outer * 32)) + ((int)threadIdx.x)) + 261120)];
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 32; ++k_inner_outer) {
      A_shared_local[0] = A_shared[((((int)threadIdx.x) * 32) + k_inner_outer)];
      A_shared_local[1] = A_shared[(((((int)threadIdx.x) * 32) + k_inner_outer) + 1024)];
      A_shared_local[2] = A_shared[(((((int)threadIdx.x) * 32) + k_inner_outer) + 2048)];
      A_shared_local[3] = A_shared[(((((int)threadIdx.x) * 32) + k_inner_outer) + 3072)];
      A_shared_local[4] = A_shared[(((((int)threadIdx.x) * 32) + k_inner_outer) + 4096)];
      A_shared_local[5] = A_shared[(((((int)threadIdx.x) * 32) + k_inner_outer) + 5120)];
      A_shared_local[6] = A_shared[(((((int)threadIdx.x) * 32) + k_inner_outer) + 6144)];
      A_shared_local[7] = A_shared[(((((int)threadIdx.x) * 32) + k_inner_outer) + 7168)];
      compute_local[0] = (compute_local[0] + A_shared_local[0]);
      compute_local[1] = (compute_local[1] + A_shared_local[1]);
      compute_local[2] = (compute_local[2] + A_shared_local[2]);
      compute_local[3] = (compute_local[3] + A_shared_local[3]);
      compute_local[4] = (compute_local[4] + A_shared_local[4]);
      compute_local[5] = (compute_local[5] + A_shared_local[5]);
      compute_local[6] = (compute_local[6] + A_shared_local[6]);
      compute_local[7] = (compute_local[7] + A_shared_local[7]);
    }
  }
  compute[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] = compute_local[0];
  compute[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 32)] = compute_local[1];
  compute[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 64)] = compute_local[2];
  compute[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 96)] = compute_local[3];
  compute[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 128)] = compute_local[4];
  compute[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 160)] = compute_local[5];
  compute[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 192)] = compute_local[6];
  compute[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 224)] = compute_local[7];
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