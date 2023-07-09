//1_1_1_1_1_1
//128_1024_14_14_256_1_1_SAME
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
extern "C" __global__ void __launch_bounds__(112) default_function_kernel(float* __restrict__ conv2d_nchw, float* __restrict__ input0, float* __restrict__ input1) {
  float conv2d_nchw_local[64];
  __shared__ float pad_temp_shared[1792];
  __shared__ float input1_shared[1024];
  for (int nn_c_inner_init = 0; nn_c_inner_init < 2; ++nn_c_inner_init) {
    for (int ff_c_inner_init = 0; ff_c_inner_init < 8; ++ff_c_inner_init) {
      conv2d_nchw_local[((nn_c_inner_init * 8) + ff_c_inner_init)] = 0.000000e+00f;
      conv2d_nchw_local[(((nn_c_inner_init * 8) + ff_c_inner_init) + 16)] = 0.000000e+00f;
      conv2d_nchw_local[(((nn_c_inner_init * 8) + ff_c_inner_init) + 32)] = 0.000000e+00f;
      conv2d_nchw_local[(((nn_c_inner_init * 8) + ff_c_inner_init) + 48)] = 0.000000e+00f;
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 64; ++rc_outer_outer) {
    __syncthreads();
    *(float2*)(pad_temp_shared + (((int)threadIdx.x) * 2)) = *(float2*)(input0 + ((((((((int)blockIdx.x) / 28) * 802816) + (rc_outer_outer * 3136)) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 14) * 2)));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 224)) = *(float2*)(input0 + (((((((((int)blockIdx.x) / 28) * 802816) + (rc_outer_outer * 3136)) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 1568));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 448)) = *(float2*)(input0 + (((((((((int)blockIdx.x) / 28) * 802816) + (rc_outer_outer * 3136)) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 200704));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 672)) = *(float2*)(input0 + (((((((((int)blockIdx.x) / 28) * 802816) + (rc_outer_outer * 3136)) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 202272));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 896)) = *(float2*)(input0 + (((((((((int)blockIdx.x) / 28) * 802816) + (rc_outer_outer * 3136)) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 401408));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 1120)) = *(float2*)(input0 + (((((((((int)blockIdx.x) / 28) * 802816) + (rc_outer_outer * 3136)) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 402976));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 1344)) = *(float2*)(input0 + (((((((((int)blockIdx.x) / 28) * 802816) + (rc_outer_outer * 3136)) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 602112));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 1568)) = *(float2*)(input0 + (((((((((int)blockIdx.x) / 28) * 802816) + (rc_outer_outer * 3136)) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 603680));
    input1_shared[((int)threadIdx.x)] = input1[((((((((int)blockIdx.x) % 28) / 7) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    input1_shared[(((int)threadIdx.x) + 112)] = input1[(((((((((int)blockIdx.x) % 28) / 7) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 7168)];
    input1_shared[(((int)threadIdx.x) + 224)] = input1[(((((((((int)blockIdx.x) % 28) / 7) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 14336)];
    input1_shared[(((int)threadIdx.x) + 336)] = input1[(((((((((int)blockIdx.x) % 28) / 7) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 21504)];
    input1_shared[(((int)threadIdx.x) + 448)] = input1[(((((((((int)blockIdx.x) % 28) / 7) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 28672)];
    input1_shared[(((int)threadIdx.x) + 560)] = input1[(((((((((int)blockIdx.x) % 28) / 7) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 35840)];
    input1_shared[(((int)threadIdx.x) + 672)] = input1[(((((((((int)blockIdx.x) % 28) / 7) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 43008)];
    input1_shared[(((int)threadIdx.x) + 784)] = input1[(((((((((int)blockIdx.x) % 28) / 7) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 50176)];
    input1_shared[(((int)threadIdx.x) + 896)] = input1[(((((((((int)blockIdx.x) % 28) / 7) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 57344)];
    if (((int)threadIdx.x) < 16) {
      input1_shared[(((int)threadIdx.x) + 1008)] = input1[((((((((int)blockIdx.x) % 28) / 7) * 65536) + (rc_outer_outer * 16)) + ((int)threadIdx.x)) + 64512)];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
        for (int nn_c_inner = 0; nn_c_inner < 2; ++nn_c_inner) {
          for (int ff_c_inner = 0; ff_c_inner < 8; ++ff_c_inner) {
            conv2d_nchw_local[((nn_c_inner * 8) + ff_c_inner)] = (conv2d_nchw_local[((nn_c_inner * 8) + ff_c_inner)] + (pad_temp_shared[((((nn_c_inner * 448) + (rc_outer_inner * 112)) + (rc_inner * 28)) + (((int)threadIdx.x) % 28))] * input1_shared[(((((((int)threadIdx.x) / 28) * 128) + (ff_c_inner * 16)) + (rc_outer_inner * 4)) + rc_inner)]));
            conv2d_nchw_local[(((nn_c_inner * 8) + ff_c_inner) + 16)] = (conv2d_nchw_local[(((nn_c_inner * 8) + ff_c_inner) + 16)] + (pad_temp_shared[((((nn_c_inner * 448) + (rc_outer_inner * 112)) + (rc_inner * 28)) + (((int)threadIdx.x) % 28))] * input1_shared[((((((((int)threadIdx.x) / 28) * 128) + (ff_c_inner * 16)) + (rc_outer_inner * 4)) + rc_inner) + 512)]));
            conv2d_nchw_local[(((nn_c_inner * 8) + ff_c_inner) + 32)] = (conv2d_nchw_local[(((nn_c_inner * 8) + ff_c_inner) + 32)] + (pad_temp_shared[(((((nn_c_inner * 448) + (rc_outer_inner * 112)) + (rc_inner * 28)) + (((int)threadIdx.x) % 28)) + 896)] * input1_shared[(((((((int)threadIdx.x) / 28) * 128) + (ff_c_inner * 16)) + (rc_outer_inner * 4)) + rc_inner)]));
            conv2d_nchw_local[(((nn_c_inner * 8) + ff_c_inner) + 48)] = (conv2d_nchw_local[(((nn_c_inner * 8) + ff_c_inner) + 48)] + (pad_temp_shared[(((((nn_c_inner * 448) + (rc_outer_inner * 112)) + (rc_inner * 28)) + (((int)threadIdx.x) % 28)) + 896)] * input1_shared[((((((((int)threadIdx.x) / 28) * 128) + (ff_c_inner * 16)) + (rc_outer_inner * 4)) + rc_inner) + 512)]));
          }
        }
      }
    }
  }
  for (int nn_inner = 0; nn_inner < 2; ++nn_inner) {
    for (int ff_inner = 0; ff_inner < 8; ++ff_inner) {
      conv2d_nchw[((((((((((int)blockIdx.x) / 28) * 200704) + (nn_inner * 50176)) + (((((int)blockIdx.x) % 28) / 7) * 12544)) + ((((int)threadIdx.x) / 28) * 1568)) + (ff_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28))] = conv2d_nchw_local[((nn_inner * 8) + ff_inner)];
      conv2d_nchw[(((((((((((int)blockIdx.x) / 28) * 200704) + (nn_inner * 50176)) + (((((int)blockIdx.x) % 28) / 7) * 12544)) + ((((int)threadIdx.x) / 28) * 1568)) + (ff_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 6272)] = conv2d_nchw_local[(((nn_inner * 8) + ff_inner) + 16)];
      conv2d_nchw[(((((((((((int)blockIdx.x) / 28) * 200704) + (nn_inner * 50176)) + (((((int)blockIdx.x) % 28) / 7) * 12544)) + ((((int)threadIdx.x) / 28) * 1568)) + (ff_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 100352)] = conv2d_nchw_local[(((nn_inner * 8) + ff_inner) + 32)];
      conv2d_nchw[(((((((((((int)blockIdx.x) / 28) * 200704) + (nn_inner * 50176)) + (((((int)blockIdx.x) % 28) / 7) * 12544)) + ((((int)threadIdx.x) / 28) * 1568)) + (ff_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 106624)] = conv2d_nchw_local[(((nn_inner * 8) + ff_inner) + 48)];
    }
  }
}

