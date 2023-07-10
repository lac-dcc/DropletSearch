//1_1_1_1_1_1
//avg_128_168_83_83_1_2_VALID
//dim3 grid(1, 1, 1);
//dim3 block(1, 1, 1);

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
extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ data, float* __restrict__ pool_avg) {
  float pool_sum[1];
  pool_sum[0] = 0.000000e+00f;
  pool_sum[0] = (pool_sum[0] + data[((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 441) * 6889) + (((((((int)blockIdx.x) * 512) + (((int)threadIdx.x) >> 1)) % 882) / 21) * 166)) + ((((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) % 42) * 2))]);
  pool_avg[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = pool_sum[0];
}

