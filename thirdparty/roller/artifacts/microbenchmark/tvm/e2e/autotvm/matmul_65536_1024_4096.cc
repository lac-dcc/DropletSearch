//2048_32_1_8_8_1
//65536_1024_4096
//dim3 grid(2048, 32, 1);
//dim3 block(8, 8, 1);

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
extern "C" __global__ void __launch_bounds__(64) matmul_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[64];
  __shared__ float placeholder_shared[1024];
  __shared__ float placeholder_d_shared[4096];
  float placeholder_shared_local[8];
  float placeholder_d_shared_local[32];
  float placeholder_shared_local1[8];
  float placeholder_d_shared_local1[32];
  for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 16; ++j_c_init) {
      T_matmul_NN_local[(((i_c_init * 16) + j_c_init))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 16) + j_c_init) + 32))] = 0.000000e+00f;
    }
  }
  for (int ax0_inner = 0; ax0_inner < 4; ++ax0_inner) {
    for (int ax1_inner_inner = 0; ax1_inner_inner < 4; ++ax1_inner_inner) {
      if (((((int)threadIdx.x) * 4) + ax1_inner_inner) < 16) {
        placeholder_shared[(((((((int)threadIdx.y) * 64) + (ax0_inner * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))] = placeholder[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 4096)) + (ax0_inner * 1024)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))];
      }
    }
  }
  for (int ax0_inner1 = 0; ax0_inner1 < 2; ++ax0_inner1) {
    for (int ax1_outer = 0; ax1_outer < 4; ++ax1_outer) {
      for (int ax1_inner_inner1 = 0; ax1_inner_inner1 < 4; ++ax1_inner_inner1) {
        placeholder_d_shared[((((((((int)threadIdx.y) * 256) + (ax0_inner1 * 128)) + (ax1_outer * 32)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))] = placeholder1[(((((((((int)threadIdx.y) * 8192) + (ax0_inner1 * 4096)) + (((int)blockIdx.y) * 128)) + (ax1_outer * 32)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))];
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 63; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_inner2 = 0; ax0_inner2 < 4; ++ax0_inner2) {
      for (int ax1_inner_inner2 = 0; ax1_inner_inner2 < 4; ++ax1_inner_inner2) {
        if (((((int)threadIdx.x) * 4) + ax1_inner_inner2) < 16) {
          if ((((k_outer_outer * 16) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2) < 1008) {
            placeholder_shared[((((((((k_outer_outer + 1) & 1) * 512) + (((int)threadIdx.y) * 64)) + (ax0_inner2 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2))] = placeholder[((((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 4096)) + (ax0_inner2 * 1024)) + (k_outer_outer * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2) + 16))];
          }
        }
      }
    }
    for (int ax0_inner3 = 0; ax0_inner3 < 2; ++ax0_inner3) {
      for (int ax1_outer1 = 0; ax1_outer1 < 4; ++ax1_outer1) {
        for (int ax1_inner_inner3 = 0; ax1_inner_inner3 < 4; ++ax1_inner_inner3) {
          placeholder_d_shared[(((((((((k_outer_outer + 1) & 1) * 2048) + (((int)threadIdx.y) * 256)) + (ax0_inner3 * 128)) + (ax1_outer1 * 32)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3))] = placeholder1[(((((((((k_outer_outer * 65536) + (((int)threadIdx.y) * 8192)) + (ax0_inner3 * 4096)) + (((int)blockIdx.y) * 128)) + (ax1_outer1 * 32)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3) + 65536))];
        }
      }
    }
    for (int ax0 = 0; ax0 < 2; ++ax0) {
      for (int ax1 = 0; ax1 < 2; ++ax1) {
        placeholder_shared_local[(((ax0 * 2) + ax1))] = placeholder_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 32)) + (ax0 * 16)) + ax1))];
        placeholder_shared_local[((((ax0 * 2) + ax1) + 4))] = placeholder_shared[(((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 32)) + (ax0 * 16)) + ax1) + 256))];
      }
    }
    for (int ax01 = 0; ax01 < 2; ++ax01) {
      for (int ax11 = 0; ax11 < 16; ++ax11) {
        placeholder_d_shared_local[(((ax01 * 16) + ax11))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (ax01 * 128)) + (((int)threadIdx.y) * 16)) + ax11))];
      }
    }
    for (int k_inner_inner = 0; k_inner_inner < 2; ++k_inner_inner) {
      for (int i_c = 0; i_c < 2; ++i_c) {
        for (int j_c = 0; j_c < 16; ++j_c) {
          T_matmul_NN_local[(((i_c * 16) + j_c))] = (T_matmul_NN_local[(((i_c * 16) + j_c))] + (placeholder_shared_local[(((i_c * 2) + k_inner_inner))] * placeholder_d_shared_local[(((k_inner_inner * 16) + j_c))]));
          T_matmul_NN_local[((((i_c * 16) + j_c) + 32))] = (T_matmul_NN_local[((((i_c * 16) + j_c) + 32))] + (placeholder_shared_local[((((i_c * 2) + k_inner_inner) + 4))] * placeholder_d_shared_local[(((k_inner_inner * 16) + j_c))]));
        }
      }
    }
    for (int ax02 = 0; ax02 < 2; ++ax02) {
      for (int ax12 = 0; ax12 < 2; ++ax12) {
        placeholder_shared_local[(((ax02 * 2) + ax12))] = placeholder_shared[(((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 32)) + (ax02 * 16)) + ax12) + 2))];
        placeholder_shared_local[((((ax02 * 2) + ax12) + 4))] = placeholder_shared[(((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 32)) + (ax02 * 16)) + ax12) + 258))];
      }
    }
    for (int ax03 = 0; ax03 < 2; ++ax03) {
      for (int ax13 = 0; ax13 < 16; ++ax13) {
        placeholder_d_shared_local[(((ax03 * 16) + ax13))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 2048) + (ax03 * 128)) + (((int)threadIdx.y) * 16)) + ax13) + 256))];
      }
    }
    for (int k_inner_inner1 = 0; k_inner_inner1 < 2; ++k_inner_inner1) {
      for (int i_c1 = 0; i_c1 < 2; ++i_c1) {
        for (int j_c1 = 0; j_c1 < 16; ++j_c1) {
          T_matmul_NN_local[(((i_c1 * 16) + j_c1))] = (T_matmul_NN_local[(((i_c1 * 16) + j_c1))] + (placeholder_shared_local[(((i_c1 * 2) + k_inner_inner1))] * placeholder_d_shared_local[(((k_inner_inner1 * 16) + j_c1))]));
          T_matmul_NN_local[((((i_c1 * 16) + j_c1) + 32))] = (T_matmul_NN_local[((((i_c1 * 16) + j_c1) + 32))] + (placeholder_shared_local[((((i_c1 * 2) + k_inner_inner1) + 4))] * placeholder_d_shared_local[(((k_inner_inner1 * 16) + j_c1))]));
        }
      }
    }
    for (int ax04 = 0; ax04 < 2; ++ax04) {
      for (int ax14 = 0; ax14 < 2; ++ax14) {
        placeholder_shared_local[(((ax04 * 2) + ax14))] = placeholder_shared[(((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 32)) + (ax04 * 16)) + ax14) + 4))];
        placeholder_shared_local[((((ax04 * 2) + ax14) + 4))] = placeholder_shared[(((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 32)) + (ax04 * 16)) + ax14) + 260))];
      }
    }
    for (int ax05 = 0; ax05 < 2; ++ax05) {
      for (int ax15 = 0; ax15 < 16; ++ax15) {
        placeholder_d_shared_local[(((ax05 * 16) + ax15))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 2048) + (ax05 * 128)) + (((int)threadIdx.y) * 16)) + ax15) + 512))];
      }
    }
    for (int k_inner_inner2 = 0; k_inner_inner2 < 2; ++k_inner_inner2) {
      for (int i_c2 = 0; i_c2 < 2; ++i_c2) {
        for (int j_c2 = 0; j_c2 < 16; ++j_c2) {
          T_matmul_NN_local[(((i_c2 * 16) + j_c2))] = (T_matmul_NN_local[(((i_c2 * 16) + j_c2))] + (placeholder_shared_local[(((i_c2 * 2) + k_inner_inner2))] * placeholder_d_shared_local[(((k_inner_inner2 * 16) + j_c2))]));
          T_matmul_NN_local[((((i_c2 * 16) + j_c2) + 32))] = (T_matmul_NN_local[((((i_c2 * 16) + j_c2) + 32))] + (placeholder_shared_local[((((i_c2 * 2) + k_inner_inner2) + 4))] * placeholder_d_shared_local[(((k_inner_inner2 * 16) + j_c2))]));
        }
      }
    }
    for (int ax06 = 0; ax06 < 2; ++ax06) {
      for (int ax16 = 0; ax16 < 2; ++ax16) {
        placeholder_shared_local[(((ax06 * 2) + ax16))] = placeholder_shared[(((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 32)) + (ax06 * 16)) + ax16) + 6))];
        placeholder_shared_local[((((ax06 * 2) + ax16) + 4))] = placeholder_shared[(((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 32)) + (ax06 * 16)) + ax16) + 262))];
      }
    }
    for (int ax07 = 0; ax07 < 2; ++ax07) {
      for (int ax17 = 0; ax17 < 16; ++ax17) {
        placeholder_d_shared_local[(((ax07 * 16) + ax17))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 2048) + (ax07 * 128)) + (((int)threadIdx.y) * 16)) + ax17) + 768))];
      }
    }
    for (int k_inner_inner3 = 0; k_inner_inner3 < 2; ++k_inner_inner3) {
      for (int i_c3 = 0; i_c3 < 2; ++i_c3) {
        for (int j_c3 = 0; j_c3 < 16; ++j_c3) {
          T_matmul_NN_local[(((i_c3 * 16) + j_c3))] = (T_matmul_NN_local[(((i_c3 * 16) + j_c3))] + (placeholder_shared_local[(((i_c3 * 2) + k_inner_inner3))] * placeholder_d_shared_local[(((k_inner_inner3 * 16) + j_c3))]));
          T_matmul_NN_local[((((i_c3 * 16) + j_c3) + 32))] = (T_matmul_NN_local[((((i_c3 * 16) + j_c3) + 32))] + (placeholder_shared_local[((((i_c3 * 2) + k_inner_inner3) + 4))] * placeholder_d_shared_local[(((k_inner_inner3 * 16) + j_c3))]));
        }
      }
    }
    for (int ax08 = 0; ax08 < 2; ++ax08) {
      for (int ax18 = 0; ax18 < 2; ++ax18) {
        placeholder_shared_local[(((ax08 * 2) + ax18))] = placeholder_shared[(((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 32)) + (ax08 * 16)) + ax18) + 8))];
        placeholder_shared_local[((((ax08 * 2) + ax18) + 4))] = placeholder_shared[(((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 32)) + (ax08 * 16)) + ax18) + 264))];
      }
    }
    for (int ax09 = 0; ax09 < 2; ++ax09) {
      for (int ax19 = 0; ax19 < 16; ++ax19) {
        placeholder_d_shared_local[(((ax09 * 16) + ax19))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 2048) + (ax09 * 128)) + (((int)threadIdx.y) * 16)) + ax19) + 1024))];
      }
    }
    for (int k_inner_inner4 = 0; k_inner_inner4 < 2; ++k_inner_inner4) {
      for (int i_c4 = 0; i_c4 < 2; ++i_c4) {
        for (int j_c4 = 0; j_c4 < 16; ++j_c4) {
          T_matmul_NN_local[(((i_c4 * 16) + j_c4))] = (T_matmul_NN_local[(((i_c4 * 16) + j_c4))] + (placeholder_shared_local[(((i_c4 * 2) + k_inner_inner4))] * placeholder_d_shared_local[(((k_inner_inner4 * 16) + j_c4))]));
          T_matmul_NN_local[((((i_c4 * 16) + j_c4) + 32))] = (T_matmul_NN_local[((((i_c4 * 16) + j_c4) + 32))] + (placeholder_shared_local[((((i_c4 * 2) + k_inner_inner4) + 4))] * placeholder_d_shared_local[(((k_inner_inner4 * 16) + j_c4))]));
        }
      }
    }
    for (int ax010 = 0; ax010 < 2; ++ax010) {
      for (int ax110 = 0; ax110 < 2; ++ax110) {
        placeholder_shared_local[(((ax010 * 2) + ax110))] = placeholder_shared[(((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 32)) + (ax010 * 16)) + ax110) + 10))];
        placeholder_shared_local[((((ax010 * 2) + ax110) + 4))] = placeholder_shared[(((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 32)) + (ax010 * 16)) + ax110) + 266))];
      }
    }
    for (int ax011 = 0; ax011 < 2; ++ax011) {
      for (int ax111 = 0; ax111 < 16; ++ax111) {
        placeholder_d_shared_local[(((ax011 * 16) + ax111))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 2048) + (ax011 * 128)) + (((int)threadIdx.y) * 16)) + ax111) + 1280))];
      }
    }
    for (int k_inner_inner5 = 0; k_inner_inner5 < 2; ++k_inner_inner5) {
      for (int i_c5 = 0; i_c5 < 2; ++i_c5) {
        for (int j_c5 = 0; j_c5 < 16; ++j_c5) {
          T_matmul_NN_local[(((i_c5 * 16) + j_c5))] = (T_matmul_NN_local[(((i_c5 * 16) + j_c5))] + (placeholder_shared_local[(((i_c5 * 2) + k_inner_inner5))] * placeholder_d_shared_local[(((k_inner_inner5 * 16) + j_c5))]));
          T_matmul_NN_local[((((i_c5 * 16) + j_c5) + 32))] = (T_matmul_NN_local[((((i_c5 * 16) + j_c5) + 32))] + (placeholder_shared_local[((((i_c5 * 2) + k_inner_inner5) + 4))] * placeholder_d_shared_local[(((k_inner_inner5 * 16) + j_c5))]));
        }
      }
    }
    for (int ax012 = 0; ax012 < 2; ++ax012) {
      for (int ax112 = 0; ax112 < 2; ++ax112) {
        placeholder_shared_local[(((ax012 * 2) + ax112))] = placeholder_shared[(((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 32)) + (ax012 * 16)) + ax112) + 12))];
        placeholder_shared_local[((((ax012 * 2) + ax112) + 4))] = placeholder_shared[(((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 32)) + (ax012 * 16)) + ax112) + 268))];
      }
    }
    for (int ax013 = 0; ax013 < 2; ++ax013) {
      for (int ax113 = 0; ax113 < 16; ++ax113) {
        placeholder_d_shared_local[(((ax013 * 16) + ax113))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 2048) + (ax013 * 128)) + (((int)threadIdx.y) * 16)) + ax113) + 1536))];
      }
    }
    for (int k_inner_inner6 = 0; k_inner_inner6 < 2; ++k_inner_inner6) {
      for (int i_c6 = 0; i_c6 < 2; ++i_c6) {
        for (int j_c6 = 0; j_c6 < 16; ++j_c6) {
          T_matmul_NN_local[(((i_c6 * 16) + j_c6))] = (T_matmul_NN_local[(((i_c6 * 16) + j_c6))] + (placeholder_shared_local[(((i_c6 * 2) + k_inner_inner6))] * placeholder_d_shared_local[(((k_inner_inner6 * 16) + j_c6))]));
          T_matmul_NN_local[((((i_c6 * 16) + j_c6) + 32))] = (T_matmul_NN_local[((((i_c6 * 16) + j_c6) + 32))] + (placeholder_shared_local[((((i_c6 * 2) + k_inner_inner6) + 4))] * placeholder_d_shared_local[(((k_inner_inner6 * 16) + j_c6))]));
        }
      }
    }
    for (int ax014 = 0; ax014 < 2; ++ax014) {
      for (int ax114 = 0; ax114 < 2; ++ax114) {
        placeholder_shared_local[(((ax014 * 2) + ax114))] = placeholder_shared[(((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 32)) + (ax014 * 16)) + ax114) + 14))];
        placeholder_shared_local[((((ax014 * 2) + ax114) + 4))] = placeholder_shared[(((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 32)) + (ax014 * 16)) + ax114) + 270))];
      }
    }
    for (int ax015 = 0; ax015 < 2; ++ax015) {
      for (int ax115 = 0; ax115 < 16; ++ax115) {
        placeholder_d_shared_local[(((ax015 * 16) + ax115))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 2048) + (ax015 * 128)) + (((int)threadIdx.y) * 16)) + ax115) + 1792))];
      }
    }
    for (int k_inner_inner7 = 0; k_inner_inner7 < 2; ++k_inner_inner7) {
      for (int i_c7 = 0; i_c7 < 2; ++i_c7) {
        for (int j_c7 = 0; j_c7 < 16; ++j_c7) {
          T_matmul_NN_local[(((i_c7 * 16) + j_c7))] = (T_matmul_NN_local[(((i_c7 * 16) + j_c7))] + (placeholder_shared_local[(((i_c7 * 2) + k_inner_inner7))] * placeholder_d_shared_local[(((k_inner_inner7 * 16) + j_c7))]));
          T_matmul_NN_local[((((i_c7 * 16) + j_c7) + 32))] = (T_matmul_NN_local[((((i_c7 * 16) + j_c7) + 32))] + (placeholder_shared_local[((((i_c7 * 2) + k_inner_inner7) + 4))] * placeholder_d_shared_local[(((k_inner_inner7 * 16) + j_c7))]));
        }
      }
    }
  }
  __syncthreads();
  for (int ax016 = 0; ax016 < 2; ++ax016) {
    for (int ax116 = 0; ax116 < 2; ++ax116) {
      placeholder_shared_local1[(((ax016 * 2) + ax116))] = placeholder_shared[(((((((int)threadIdx.x) * 32) + (ax016 * 16)) + ax116) + 512))];
      placeholder_shared_local1[((((ax016 * 2) + ax116) + 4))] = placeholder_shared[(((((((int)threadIdx.x) * 32) + (ax016 * 16)) + ax116) + 768))];
    }
  }
  for (int ax017 = 0; ax017 < 2; ++ax017) {
    for (int ax117 = 0; ax117 < 16; ++ax117) {
      placeholder_d_shared_local1[(((ax017 * 16) + ax117))] = placeholder_d_shared[(((((ax017 * 128) + (((int)threadIdx.y) * 16)) + ax117) + 2048))];
    }
  }
  for (int k_inner_inner8 = 0; k_inner_inner8 < 2; ++k_inner_inner8) {
    for (int i_c8 = 0; i_c8 < 2; ++i_c8) {
      for (int j_c8 = 0; j_c8 < 16; ++j_c8) {
        T_matmul_NN_local[(((i_c8 * 16) + j_c8))] = (T_matmul_NN_local[(((i_c8 * 16) + j_c8))] + (placeholder_shared_local1[(((i_c8 * 2) + k_inner_inner8))] * placeholder_d_shared_local1[(((k_inner_inner8 * 16) + j_c8))]));
        T_matmul_NN_local[((((i_c8 * 16) + j_c8) + 32))] = (T_matmul_NN_local[((((i_c8 * 16) + j_c8) + 32))] + (placeholder_shared_local1[((((i_c8 * 2) + k_inner_inner8) + 4))] * placeholder_d_shared_local1[(((k_inner_inner8 * 16) + j_c8))]));
      }
    }
  }
  for (int ax018 = 0; ax018 < 2; ++ax018) {
    for (int ax118 = 0; ax118 < 2; ++ax118) {
      placeholder_shared_local1[(((ax018 * 2) + ax118))] = placeholder_shared[(((((((int)threadIdx.x) * 32) + (ax018 * 16)) + ax118) + 514))];
      placeholder_shared_local1[((((ax018 * 2) + ax118) + 4))] = placeholder_shared[(((((((int)threadIdx.x) * 32) + (ax018 * 16)) + ax118) + 770))];
    }
  }
  for (int ax019 = 0; ax019 < 2; ++ax019) {
    for (int ax119 = 0; ax119 < 16; ++ax119) {
      placeholder_d_shared_local1[(((ax019 * 16) + ax119))] = placeholder_d_shared[(((((ax019 * 128) + (((int)threadIdx.y) * 16)) + ax119) + 2304))];
    }
  }
  for (int k_inner_inner9 = 0; k_inner_inner9 < 2; ++k_inner_inner9) {
    for (int i_c9 = 0; i_c9 < 2; ++i_c9) {
      for (int j_c9 = 0; j_c9 < 16; ++j_c9) {
        T_matmul_NN_local[(((i_c9 * 16) + j_c9))] = (T_matmul_NN_local[(((i_c9 * 16) + j_c9))] + (placeholder_shared_local1[(((i_c9 * 2) + k_inner_inner9))] * placeholder_d_shared_local1[(((k_inner_inner9 * 16) + j_c9))]));
        T_matmul_NN_local[((((i_c9 * 16) + j_c9) + 32))] = (T_matmul_NN_local[((((i_c9 * 16) + j_c9) + 32))] + (placeholder_shared_local1[((((i_c9 * 2) + k_inner_inner9) + 4))] * placeholder_d_shared_local1[(((k_inner_inner9 * 16) + j_c9))]));
      }
    }
  }
  for (int ax020 = 0; ax020 < 2; ++ax020) {
    for (int ax120 = 0; ax120 < 2; ++ax120) {
      placeholder_shared_local1[(((ax020 * 2) + ax120))] = placeholder_shared[(((((((int)threadIdx.x) * 32) + (ax020 * 16)) + ax120) + 516))];
      placeholder_shared_local1[((((ax020 * 2) + ax120) + 4))] = placeholder_shared[(((((((int)threadIdx.x) * 32) + (ax020 * 16)) + ax120) + 772))];
    }
  }
  for (int ax021 = 0; ax021 < 2; ++ax021) {
    for (int ax121 = 0; ax121 < 16; ++ax121) {
      placeholder_d_shared_local1[(((ax021 * 16) + ax121))] = placeholder_d_shared[(((((ax021 * 128) + (((int)threadIdx.y) * 16)) + ax121) + 2560))];
    }
  }
  for (int k_inner_inner10 = 0; k_inner_inner10 < 2; ++k_inner_inner10) {
    for (int i_c10 = 0; i_c10 < 2; ++i_c10) {
      for (int j_c10 = 0; j_c10 < 16; ++j_c10) {
        T_matmul_NN_local[(((i_c10 * 16) + j_c10))] = (T_matmul_NN_local[(((i_c10 * 16) + j_c10))] + (placeholder_shared_local1[(((i_c10 * 2) + k_inner_inner10))] * placeholder_d_shared_local1[(((k_inner_inner10 * 16) + j_c10))]));
        T_matmul_NN_local[((((i_c10 * 16) + j_c10) + 32))] = (T_matmul_NN_local[((((i_c10 * 16) + j_c10) + 32))] + (placeholder_shared_local1[((((i_c10 * 2) + k_inner_inner10) + 4))] * placeholder_d_shared_local1[(((k_inner_inner10 * 16) + j_c10))]));
      }
    }
  }
  for (int ax022 = 0; ax022 < 2; ++ax022) {
    for (int ax122 = 0; ax122 < 2; ++ax122) {
      placeholder_shared_local1[(((ax022 * 2) + ax122))] = placeholder_shared[(((((((int)threadIdx.x) * 32) + (ax022 * 16)) + ax122) + 518))];
      placeholder_shared_local1[((((ax022 * 2) + ax122) + 4))] = placeholder_shared[(((((((int)threadIdx.x) * 32) + (ax022 * 16)) + ax122) + 774))];
    }
  }
  for (int ax023 = 0; ax023 < 2; ++ax023) {
    for (int ax123 = 0; ax123 < 16; ++ax123) {
      placeholder_d_shared_local1[(((ax023 * 16) + ax123))] = placeholder_d_shared[(((((ax023 * 128) + (((int)threadIdx.y) * 16)) + ax123) + 2816))];
    }
  }
  for (int k_inner_inner11 = 0; k_inner_inner11 < 2; ++k_inner_inner11) {
    for (int i_c11 = 0; i_c11 < 2; ++i_c11) {
      for (int j_c11 = 0; j_c11 < 16; ++j_c11) {
        T_matmul_NN_local[(((i_c11 * 16) + j_c11))] = (T_matmul_NN_local[(((i_c11 * 16) + j_c11))] + (placeholder_shared_local1[(((i_c11 * 2) + k_inner_inner11))] * placeholder_d_shared_local1[(((k_inner_inner11 * 16) + j_c11))]));
        T_matmul_NN_local[((((i_c11 * 16) + j_c11) + 32))] = (T_matmul_NN_local[((((i_c11 * 16) + j_c11) + 32))] + (placeholder_shared_local1[((((i_c11 * 2) + k_inner_inner11) + 4))] * placeholder_d_shared_local1[(((k_inner_inner11 * 16) + j_c11))]));
      }
    }
  }
  for (int ax024 = 0; ax024 < 2; ++ax024) {
    for (int ax124 = 0; ax124 < 2; ++ax124) {
      placeholder_shared_local1[(((ax024 * 2) + ax124))] = placeholder_shared[(((((((int)threadIdx.x) * 32) + (ax024 * 16)) + ax124) + 520))];
      placeholder_shared_local1[((((ax024 * 2) + ax124) + 4))] = placeholder_shared[(((((((int)threadIdx.x) * 32) + (ax024 * 16)) + ax124) + 776))];
    }
  }
  for (int ax025 = 0; ax025 < 2; ++ax025) {
    for (int ax125 = 0; ax125 < 16; ++ax125) {
      placeholder_d_shared_local1[(((ax025 * 16) + ax125))] = placeholder_d_shared[(((((ax025 * 128) + (((int)threadIdx.y) * 16)) + ax125) + 3072))];
    }
  }
  for (int k_inner_inner12 = 0; k_inner_inner12 < 2; ++k_inner_inner12) {
    for (int i_c12 = 0; i_c12 < 2; ++i_c12) {
      for (int j_c12 = 0; j_c12 < 16; ++j_c12) {
        T_matmul_NN_local[(((i_c12 * 16) + j_c12))] = (T_matmul_NN_local[(((i_c12 * 16) + j_c12))] + (placeholder_shared_local1[(((i_c12 * 2) + k_inner_inner12))] * placeholder_d_shared_local1[(((k_inner_inner12 * 16) + j_c12))]));
        T_matmul_NN_local[((((i_c12 * 16) + j_c12) + 32))] = (T_matmul_NN_local[((((i_c12 * 16) + j_c12) + 32))] + (placeholder_shared_local1[((((i_c12 * 2) + k_inner_inner12) + 4))] * placeholder_d_shared_local1[(((k_inner_inner12 * 16) + j_c12))]));
      }
    }
  }
  for (int ax026 = 0; ax026 < 2; ++ax026) {
    for (int ax126 = 0; ax126 < 2; ++ax126) {
      placeholder_shared_local1[(((ax026 * 2) + ax126))] = placeholder_shared[(((((((int)threadIdx.x) * 32) + (ax026 * 16)) + ax126) + 522))];
      placeholder_shared_local1[((((ax026 * 2) + ax126) + 4))] = placeholder_shared[(((((((int)threadIdx.x) * 32) + (ax026 * 16)) + ax126) + 778))];
    }
  }
  for (int ax027 = 0; ax027 < 2; ++ax027) {
    for (int ax127 = 0; ax127 < 16; ++ax127) {
      placeholder_d_shared_local1[(((ax027 * 16) + ax127))] = placeholder_d_shared[(((((ax027 * 128) + (((int)threadIdx.y) * 16)) + ax127) + 3328))];
    }
  }
  for (int k_inner_inner13 = 0; k_inner_inner13 < 2; ++k_inner_inner13) {
    for (int i_c13 = 0; i_c13 < 2; ++i_c13) {
      for (int j_c13 = 0; j_c13 < 16; ++j_c13) {
        T_matmul_NN_local[(((i_c13 * 16) + j_c13))] = (T_matmul_NN_local[(((i_c13 * 16) + j_c13))] + (placeholder_shared_local1[(((i_c13 * 2) + k_inner_inner13))] * placeholder_d_shared_local1[(((k_inner_inner13 * 16) + j_c13))]));
        T_matmul_NN_local[((((i_c13 * 16) + j_c13) + 32))] = (T_matmul_NN_local[((((i_c13 * 16) + j_c13) + 32))] + (placeholder_shared_local1[((((i_c13 * 2) + k_inner_inner13) + 4))] * placeholder_d_shared_local1[(((k_inner_inner13 * 16) + j_c13))]));
      }
    }
  }
  for (int ax028 = 0; ax028 < 2; ++ax028) {
    for (int ax128 = 0; ax128 < 2; ++ax128) {
      placeholder_shared_local1[(((ax028 * 2) + ax128))] = placeholder_shared[(((((((int)threadIdx.x) * 32) + (ax028 * 16)) + ax128) + 524))];
      placeholder_shared_local1[((((ax028 * 2) + ax128) + 4))] = placeholder_shared[(((((((int)threadIdx.x) * 32) + (ax028 * 16)) + ax128) + 780))];
    }
  }
  for (int ax029 = 0; ax029 < 2; ++ax029) {
    for (int ax129 = 0; ax129 < 16; ++ax129) {
      placeholder_d_shared_local1[(((ax029 * 16) + ax129))] = placeholder_d_shared[(((((ax029 * 128) + (((int)threadIdx.y) * 16)) + ax129) + 3584))];
    }
  }
  for (int k_inner_inner14 = 0; k_inner_inner14 < 2; ++k_inner_inner14) {
    for (int i_c14 = 0; i_c14 < 2; ++i_c14) {
      for (int j_c14 = 0; j_c14 < 16; ++j_c14) {
        T_matmul_NN_local[(((i_c14 * 16) + j_c14))] = (T_matmul_NN_local[(((i_c14 * 16) + j_c14))] + (placeholder_shared_local1[(((i_c14 * 2) + k_inner_inner14))] * placeholder_d_shared_local1[(((k_inner_inner14 * 16) + j_c14))]));
        T_matmul_NN_local[((((i_c14 * 16) + j_c14) + 32))] = (T_matmul_NN_local[((((i_c14 * 16) + j_c14) + 32))] + (placeholder_shared_local1[((((i_c14 * 2) + k_inner_inner14) + 4))] * placeholder_d_shared_local1[(((k_inner_inner14 * 16) + j_c14))]));
      }
    }
  }
  for (int ax030 = 0; ax030 < 2; ++ax030) {
    for (int ax130 = 0; ax130 < 2; ++ax130) {
      placeholder_shared_local1[(((ax030 * 2) + ax130))] = placeholder_shared[(((((((int)threadIdx.x) * 32) + (ax030 * 16)) + ax130) + 526))];
      placeholder_shared_local1[((((ax030 * 2) + ax130) + 4))] = placeholder_shared[(((((((int)threadIdx.x) * 32) + (ax030 * 16)) + ax130) + 782))];
    }
  }
  for (int ax031 = 0; ax031 < 2; ++ax031) {
    for (int ax131 = 0; ax131 < 16; ++ax131) {
      placeholder_d_shared_local1[(((ax031 * 16) + ax131))] = placeholder_d_shared[(((((ax031 * 128) + (((int)threadIdx.y) * 16)) + ax131) + 3840))];
    }
  }
  for (int k_inner_inner15 = 0; k_inner_inner15 < 2; ++k_inner_inner15) {
    for (int i_c15 = 0; i_c15 < 2; ++i_c15) {
      for (int j_c15 = 0; j_c15 < 16; ++j_c15) {
        T_matmul_NN_local[(((i_c15 * 16) + j_c15))] = (T_matmul_NN_local[(((i_c15 * 16) + j_c15))] + (placeholder_shared_local1[(((i_c15 * 2) + k_inner_inner15))] * placeholder_d_shared_local1[(((k_inner_inner15 * 16) + j_c15))]));
        T_matmul_NN_local[((((i_c15 * 16) + j_c15) + 32))] = (T_matmul_NN_local[((((i_c15 * 16) + j_c15) + 32))] + (placeholder_shared_local1[((((i_c15 * 2) + k_inner_inner15) + 4))] * placeholder_d_shared_local1[(((k_inner_inner15 * 16) + j_c15))]));
      }
    }
  }
  for (int j_inner_inner_inner = 0; j_inner_inner_inner < 16; ++j_inner_inner_inner) {
    for (int i_inner_inner_inner = 0; i_inner_inner_inner < 2; ++i_inner_inner_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 8192)) + (i_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 16)) + j_inner_inner_inner))] = T_matmul_NN_local[(((i_inner_inner_inner * 16) + j_inner_inner_inner))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 8192)) + (i_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 16)) + j_inner_inner_inner) + 65536))] = T_matmul_NN_local[((((i_inner_inner_inner * 16) + j_inner_inner_inner) + 32))];
    }
  }
}

