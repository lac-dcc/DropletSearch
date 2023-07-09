//1_1_1_1_1_1
//128_168_83_83_84_1_1_SAME
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
extern "C" __global__ void __launch_bounds__(249) default_function_kernel(float* __restrict__ conv2d_nchw, float* __restrict__ input0, float* __restrict__ input1) {
  float conv2d_nchw_local[112];
  __shared__ float pad_temp_shared[7968];
  __shared__ float input1_shared[2016];
  for (int ff_c_outer_inner_init = 0; ff_c_outer_inner_init < 28; ++ff_c_outer_inner_init) {
    for (int nn_c_inner_init = 0; nn_c_inner_init < 4; ++nn_c_inner_init) {
      conv2d_nchw_local[((nn_c_inner_init * 28) + ff_c_outer_inner_init)] = 0.000000e+00f;
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 7; ++rc_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 32; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      pad_temp_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 249) + ((int)threadIdx.x))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer >> 3) * 1157352)) + (rc_outer_outer * 165336)) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer & 7) * 20667)) + ((((int)threadIdx.x) / 83) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 * 83) + (((int)threadIdx.x) / 3)) < 672) {
        input1_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 * 249) + ((int)threadIdx.x))] = input1[((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 * 83) + (((int)threadIdx.x) / 3)) >> 3) * 168) + (rc_outer_outer * 24)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 * 9) + ((int)threadIdx.x)) % 24))];
      }
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 12; ++rc_outer_inner) {
      for (int ff_c_outer_inner = 0; ff_c_outer_inner < 28; ++ff_c_outer_inner) {
        for (int rc_inner = 0; rc_inner < 2; ++rc_inner) {
          for (int nn_c_inner = 0; nn_c_inner < 4; ++nn_c_inner) {
            conv2d_nchw_local[((nn_c_inner * 28) + ff_c_outer_inner)] = (conv2d_nchw_local[((nn_c_inner * 28) + ff_c_outer_inner)] + (pad_temp_shared[((((nn_c_inner * 1992) + (rc_outer_inner * 166)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83))] * input1_shared[(((((((int)threadIdx.x) / 83) * 672) + (ff_c_outer_inner * 24)) + (rc_outer_inner * 2)) + rc_inner)]));
          }
        }
      }
    }
  }
  for (int nn_inner = 0; nn_inner < 4; ++nn_inner) {
    for (int ff_inner = 0; ff_inner < 28; ++ff_inner) {
      conv2d_nchw[(((((((((int)blockIdx.x) / 83) * 2314704) + (nn_inner * 578676)) + ((((int)threadIdx.x) / 83) * 192892)) + (ff_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83))] = conv2d_nchw_local[((nn_inner * 28) + ff_inner)];
    }
  }
}

