//1_1_2048_7_2_8
//128_256_14_14_256_3_1_SAME
//dim3 grid(1, 1, 2048);
//dim3 block(7, 2, 8);

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
extern "C" __global__ void __launch_bounds__(112) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute, float* __restrict__ input2) {
  float compute1[28];
  __shared__ float pad_temp_shared[256];
  __shared__ float placeholder_shared[144];
  compute1[(0)] = 0.000000e+00f;
  compute1[(1)] = 0.000000e+00f;
  compute1[(2)] = 0.000000e+00f;
  compute1[(3)] = 0.000000e+00f;
  compute1[(4)] = 0.000000e+00f;
  compute1[(5)] = 0.000000e+00f;
  compute1[(6)] = 0.000000e+00f;
  compute1[(7)] = 0.000000e+00f;
  compute1[(8)] = 0.000000e+00f;
  compute1[(9)] = 0.000000e+00f;
  compute1[(10)] = 0.000000e+00f;
  compute1[(11)] = 0.000000e+00f;
  compute1[(12)] = 0.000000e+00f;
  compute1[(13)] = 0.000000e+00f;
  compute1[(14)] = 0.000000e+00f;
  compute1[(15)] = 0.000000e+00f;
  compute1[(16)] = 0.000000e+00f;
  compute1[(17)] = 0.000000e+00f;
  compute1[(18)] = 0.000000e+00f;
  compute1[(19)] = 0.000000e+00f;
  compute1[(20)] = 0.000000e+00f;
  compute1[(21)] = 0.000000e+00f;
  compute1[(22)] = 0.000000e+00f;
  compute1[(23)] = 0.000000e+00f;
  compute1[(24)] = 0.000000e+00f;
  compute1[(25)] = 0.000000e+00f;
  compute1[(26)] = 0.000000e+00f;
  compute1[(27)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 256; ++rc_outer) {
    __syncthreads();
    if ((((((int)threadIdx.z) * 2) + ((((int)threadIdx.x) * 3) >> 4)) + ((int)threadIdx.y)) < 16) {
      if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 3)) < 256) {
        if (((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 3)) < 32) {
          if (((int)threadIdx.x) < 6) {
            pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 3)))] = (((((1 <= ((((int)threadIdx.z) * 2) + ((int)threadIdx.y))) && (((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) < 15)) && (1 <= ((int)threadIdx.x))) && (((int)threadIdx.x) < 5)) ? placeholder[((((((((((int)blockIdx.z) >> 4) * 50176) + (rc_outer * 196)) + (((int)threadIdx.z) * 28)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 3)) - 15))] : 0.000000e+00f);
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 3) + 1) >> 4)) + ((int)threadIdx.y)) < 16) {
      if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 3)) < 255) {
        if (((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 3)) < 31) {
          if (((int)threadIdx.x) < 5) {
            pad_temp_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 3)) + 1))] = (((1 <= ((((int)threadIdx.z) * 2) + ((int)threadIdx.y))) && (((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) < 15)) ? placeholder[((((((((((int)blockIdx.z) >> 4) * 50176) + (rc_outer * 196)) + (((int)threadIdx.z) * 28)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 3)) - 14))] : 0.000000e+00f);
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 3) + 2) >> 4)) + ((int)threadIdx.y)) < 16) {
      if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 3)) < 254) {
        if (((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 3)) < 30) {
          if (((int)threadIdx.x) < 5) {
            pad_temp_shared[(((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 3)) + 2))] = (((1 <= ((((int)threadIdx.z) * 2) + ((int)threadIdx.y))) && (((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) < 15)) ? placeholder[((((((((((int)blockIdx.z) >> 4) * 50176) + (rc_outer * 196)) + (((int)threadIdx.z) * 28)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 3)) - 13))] : 0.000000e+00f);
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 2) + ((((int)threadIdx.x) * 2) / 9)) + ((int)threadIdx.y)) < 16) {
      if ((((((int)threadIdx.z) * 6) + (((int)threadIdx.y) * 3)) + ((((int)threadIdx.x) * 2) / 3)) < 48) {
        if ((((((int)threadIdx.z) * 18) + (((int)threadIdx.y) * 9)) + (((int)threadIdx.x) * 2)) < 144) {
          if (((((int)threadIdx.y) * 9) + (((int)threadIdx.x) * 2)) < 18) {
            if (((int)threadIdx.x) < 5) {
              placeholder_shared[((((((int)threadIdx.z) * 18) + (((int)threadIdx.y) * 9)) + (((int)threadIdx.x) * 2)))] = placeholder1[(((((((((int)blockIdx.z) & 15) * 36864) + (((int)threadIdx.z) * 4608)) + (((int)threadIdx.y) * 2304)) + (rc_outer * 9)) + (((int)threadIdx.x) * 2)))];
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 2) + 1) / 9)) + ((int)threadIdx.y)) < 16) {
      if ((((((int)threadIdx.z) * 6) + (((int)threadIdx.y) * 3)) + (((((int)threadIdx.x) * 2) + 1) / 3)) < 48) {
        if ((((((int)threadIdx.z) * 18) + (((int)threadIdx.y) * 9)) + (((int)threadIdx.x) * 2)) < 143) {
          if (((((int)threadIdx.y) * 9) + (((int)threadIdx.x) * 2)) < 17) {
            if (((int)threadIdx.x) < 4) {
              placeholder_shared[(((((((int)threadIdx.z) * 18) + (((int)threadIdx.y) * 9)) + (((int)threadIdx.x) * 2)) + 1))] = placeholder1[((((((((((int)blockIdx.z) & 15) * 36864) + (((int)threadIdx.z) * 4608)) + (((int)threadIdx.y) * 2304)) + (rc_outer * 9)) + (((int)threadIdx.x) * 2)) + 1))];
            }
          }
        }
      }
    }
    __syncthreads();
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 1))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 16))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 17))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 32))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 33))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 48))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 49))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 64))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 65))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 80))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 81))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 96))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 97))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 48))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 64))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 65))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 80))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 96))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 97))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 18))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 50))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 65))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 66))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 82))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 97))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 18))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 50))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 65))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 66))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 82))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 97))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 18))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 35))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 50))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 51))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 66))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 67))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 82))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 83))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 99))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 18))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 35))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 50))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 51))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 66))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 67))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 82))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 83))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 99))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 48))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 64))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 65))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 80))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 96))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 97))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 113))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 48))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 64))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 65))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 80))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 96))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 97))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 113))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 18))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 50))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 65))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 66))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 82))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 97))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 113))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 114))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 18))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 50))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 65))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 66))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 82))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 97))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 113))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 114))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 18))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 35))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 50))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 51))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 66))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 67))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 82))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 83))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 99))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 114))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 115))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 18))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 35))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 50))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 51))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 66))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 67))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 82))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 83))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 99))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 114))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 115))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 48))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 64))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 65))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 80))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 96))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 97))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 113))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 128))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 129))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 48))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 64))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 65))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 80))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 96))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 97))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 113))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 128))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 129))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 50))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 65))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 66))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 82))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 97))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 113))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 114))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 129))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 50))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 65))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 66))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 82))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 97))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 113))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 114))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 129))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 35))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 50))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 51))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 66))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 67))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 82))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 83))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 99))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 114))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 115))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 131))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 34))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 35))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 50))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 51))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 66))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 67))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 82))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 83))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 99))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 114))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 115))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[((((((int)threadIdx.y) * 112) + (((int)threadIdx.x) * 2)) + 131))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
  }
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)))] = max((compute1[(0)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 1))] = max((compute1[(1)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 1))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 14))] = max((compute1[(2)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 14))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 15))] = max((compute1[(3)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 15))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 28))] = max((compute1[(4)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 28))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 29))] = max((compute1[(5)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 29))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 42))] = max((compute1[(6)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 42))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 43))] = max((compute1[(7)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 43))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 56))] = max((compute1[(8)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 56))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 57))] = max((compute1[(9)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 57))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 70))] = max((compute1[(10)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 70))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 71))] = max((compute1[(11)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 71))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 84))] = max((compute1[(12)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 84))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 85))] = max((compute1[(13)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 85))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 196))] = max((compute1[(14)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 196))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 197))] = max((compute1[(15)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 197))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 210))] = max((compute1[(16)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 210))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 211))] = max((compute1[(17)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 211))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 224))] = max((compute1[(18)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 224))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 225))] = max((compute1[(19)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 225))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 238))] = max((compute1[(20)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 238))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 239))] = max((compute1[(21)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 239))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 252))] = max((compute1[(22)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 252))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 253))] = max((compute1[(23)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 253))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 266))] = max((compute1[(24)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 266))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 267))] = max((compute1[(25)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 267))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 280))] = max((compute1[(26)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 280))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 281))] = max((compute1[(27)] + input2[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 2)) + 281))]), 0.000000e+00f);
}

