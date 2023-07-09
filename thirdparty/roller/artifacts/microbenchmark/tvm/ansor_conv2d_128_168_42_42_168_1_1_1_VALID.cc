//1_1_1_1_1_1
//128_168_42_42_168_1_1_VALID
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
extern "C" __global__ void __launch_bounds__(84) default_function_kernel(float* __restrict__ conv2d_nchw, float* __restrict__ input0, float* __restrict__ input1) {
  float conv2d_nchw_local[168];
  __shared__ float pad_temp_shared[1344];
  __shared__ float input1_shared[672];
  for (int ff_c_outer_inner_init = 0; ff_c_outer_inner_init < 6; ++ff_c_outer_inner_init) {
    for (int ff_c_inner_init = 0; ff_c_inner_init < 7; ++ff_c_inner_init) {
      conv2d_nchw_local[((ff_c_outer_inner_init * 7) + ff_c_inner_init)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 7) + ff_c_inner_init) + 42)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 7) + ff_c_inner_init) + 84)] = 0.000000e+00f;
      conv2d_nchw_local[(((ff_c_outer_inner_init * 7) + ff_c_inner_init) + 126)] = 0.000000e+00f;
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 21; ++rc_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      *(float2*)(pad_temp_shared + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 168) + (((int)threadIdx.x) * 2))) = *(float2*)(input0 + ((((((((((((int)blockIdx.x) / 42) * 592704) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer >> 2) * 296352)) + (rc_outer_outer * 14112)) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer & 3) * 3528)) + ((((int)threadIdx.x) / 42) * 1764)) + (((((int)blockIdx.x) % 21) / 7) * 588)) + (((((int)threadIdx.x) % 42) / 3) * 42)) + ((((int)blockIdx.x) % 7) * 6)) + ((((int)threadIdx.x) % 3) * 2)));
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1) {
      input1_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 * 84) + ((int)threadIdx.x))] = input1[((((((((int)blockIdx.x) % 42) / 21) * 14112) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 * 21) + (((int)threadIdx.x) >> 2)) >> 1) * 168)) + (rc_outer_outer * 8)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 * 4) + ((int)threadIdx.x)) & 7))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 8; ++rc_outer_inner) {
      for (int ff_c_outer_inner = 0; ff_c_outer_inner < 6; ++ff_c_outer_inner) {
        for (int ff_c_inner = 0; ff_c_inner < 7; ++ff_c_inner) {
          conv2d_nchw_local[((ff_c_outer_inner * 7) + ff_c_inner)] = (conv2d_nchw_local[((ff_c_outer_inner * 7) + ff_c_inner)] + (pad_temp_shared[((rc_outer_inner * 84) + (((int)threadIdx.x) % 42))] * input1_shared[(((((((int)threadIdx.x) / 42) * 336) + (ff_c_outer_inner * 56)) + (ff_c_inner * 8)) + rc_outer_inner)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 7) + ff_c_inner) + 42)] = (conv2d_nchw_local[(((ff_c_outer_inner * 7) + ff_c_inner) + 42)] + (pad_temp_shared[(((rc_outer_inner * 84) + (((int)threadIdx.x) % 42)) + 42)] * input1_shared[(((((((int)threadIdx.x) / 42) * 336) + (ff_c_outer_inner * 56)) + (ff_c_inner * 8)) + rc_outer_inner)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 7) + ff_c_inner) + 84)] = (conv2d_nchw_local[(((ff_c_outer_inner * 7) + ff_c_inner) + 84)] + (pad_temp_shared[(((rc_outer_inner * 84) + (((int)threadIdx.x) % 42)) + 672)] * input1_shared[(((((((int)threadIdx.x) / 42) * 336) + (ff_c_outer_inner * 56)) + (ff_c_inner * 8)) + rc_outer_inner)]));
          conv2d_nchw_local[(((ff_c_outer_inner * 7) + ff_c_inner) + 126)] = (conv2d_nchw_local[(((ff_c_outer_inner * 7) + ff_c_inner) + 126)] + (pad_temp_shared[(((rc_outer_inner * 84) + (((int)threadIdx.x) % 42)) + 714)] * input1_shared[(((((((int)threadIdx.x) / 42) * 336) + (ff_c_outer_inner * 56)) + (ff_c_inner * 8)) + rc_outer_inner)]));
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 42; ++ff_inner) {
    conv2d_nchw[(((((((((((int)blockIdx.x) / 42) * 592704) + (((((int)blockIdx.x) % 42) / 21) * 148176)) + ((((int)threadIdx.x) / 42) * 74088)) + (ff_inner * 1764)) + (((((int)blockIdx.x) % 21) / 7) * 588)) + (((((int)threadIdx.x) % 42) / 6) * 42)) + ((((int)blockIdx.x) % 7) * 6)) + (((int)threadIdx.x) % 6))] = conv2d_nchw_local[ff_inner];
    conv2d_nchw[((((((((((((int)blockIdx.x) / 42) * 592704) + (((((int)blockIdx.x) % 42) / 21) * 148176)) + ((((int)threadIdx.x) / 42) * 74088)) + (ff_inner * 1764)) + (((((int)blockIdx.x) % 21) / 7) * 588)) + (((((int)threadIdx.x) % 42) / 6) * 42)) + ((((int)blockIdx.x) % 7) * 6)) + (((int)threadIdx.x) % 6)) + 294)] = conv2d_nchw_local[(ff_inner + 42)];
    conv2d_nchw[((((((((((((int)blockIdx.x) / 42) * 592704) + (((((int)blockIdx.x) % 42) / 21) * 148176)) + ((((int)threadIdx.x) / 42) * 74088)) + (ff_inner * 1764)) + (((((int)blockIdx.x) % 21) / 7) * 588)) + (((((int)threadIdx.x) % 42) / 6) * 42)) + ((((int)blockIdx.x) % 7) * 6)) + (((int)threadIdx.x) % 6)) + 296352)] = conv2d_nchw_local[(ff_inner + 84)];
    conv2d_nchw[((((((((((((int)blockIdx.x) / 42) * 592704) + (((((int)blockIdx.x) % 42) / 21) * 148176)) + ((((int)threadIdx.x) / 42) * 74088)) + (ff_inner * 1764)) + (((((int)blockIdx.x) % 21) / 7) * 588)) + (((((int)threadIdx.x) % 42) / 6) * 42)) + ((((int)blockIdx.x) % 7) * 6)) + (((int)threadIdx.x) % 6)) + 296646)] = conv2d_nchw_local[(ff_inner + 126)];
  }
}

