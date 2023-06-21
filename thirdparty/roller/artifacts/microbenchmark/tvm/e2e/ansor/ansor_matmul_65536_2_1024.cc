//131072_1_1_16_1_1
//65536_2_1024
//dim3 grid(131072, 1, 1);
//dim3 block(16, 1, 1);

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
extern "C" __global__ void __launch_bounds__(16) default_function_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute) {
  float compute_local[32];
  __shared__ float A_shared[32];
  __shared__ float B_shared[64];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(16)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(17)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(18)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(19)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(20)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(21)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(22)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  compute_local[(23)] = 0.000000e+00f;
  compute_local[(8)] = 0.000000e+00f;
  compute_local[(24)] = 0.000000e+00f;
  compute_local[(9)] = 0.000000e+00f;
  compute_local[(25)] = 0.000000e+00f;
  compute_local[(10)] = 0.000000e+00f;
  compute_local[(26)] = 0.000000e+00f;
  compute_local[(11)] = 0.000000e+00f;
  compute_local[(27)] = 0.000000e+00f;
  compute_local[(12)] = 0.000000e+00f;
  compute_local[(28)] = 0.000000e+00f;
  compute_local[(13)] = 0.000000e+00f;
  compute_local[(29)] = 0.000000e+00f;
  compute_local[(14)] = 0.000000e+00f;
  compute_local[(30)] = 0.000000e+00f;
  compute_local[(15)] = 0.000000e+00f;
  compute_local[(31)] = 0.000000e+00f;
  if (((int)threadIdx.x) < 4) {
    A_shared[((((int)threadIdx.x) * 8))] = A[((((((int)blockIdx.x) >> 5) * 32) + (((int)threadIdx.x) * 8)))];
  }
  if (((int)threadIdx.x) < 4) {
    A_shared[(((((int)threadIdx.x) * 8) + 1))] = A[(((((((int)blockIdx.x) >> 5) * 32) + (((int)threadIdx.x) * 8)) + 1))];
  }
  if (((int)threadIdx.x) < 4) {
    A_shared[(((((int)threadIdx.x) * 8) + 2))] = A[(((((((int)blockIdx.x) >> 5) * 32) + (((int)threadIdx.x) * 8)) + 2))];
  }
  if (((int)threadIdx.x) < 4) {
    A_shared[(((((int)threadIdx.x) * 8) + 3))] = A[(((((((int)blockIdx.x) >> 5) * 32) + (((int)threadIdx.x) * 8)) + 3))];
  }
  if (((int)threadIdx.x) < 4) {
    A_shared[(((((int)threadIdx.x) * 8) + 4))] = A[(((((((int)blockIdx.x) >> 5) * 32) + (((int)threadIdx.x) * 8)) + 4))];
  }
  if (((int)threadIdx.x) < 4) {
    A_shared[(((((int)threadIdx.x) * 8) + 5))] = A[(((((((int)blockIdx.x) >> 5) * 32) + (((int)threadIdx.x) * 8)) + 5))];
  }
  if (((int)threadIdx.x) < 4) {
    A_shared[(((((int)threadIdx.x) * 8) + 6))] = A[(((((((int)blockIdx.x) >> 5) * 32) + (((int)threadIdx.x) * 8)) + 6))];
  }
  if (((int)threadIdx.x) < 4) {
    A_shared[(((((int)threadIdx.x) * 8) + 7))] = A[(((((((int)blockIdx.x) >> 5) * 32) + (((int)threadIdx.x) * 8)) + 7))];
  }
  ((float4*)(B_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(B + (((((((int)threadIdx.x) >> 3) * 1024) + ((((int)blockIdx.x) & 31) * 32)) + ((((int)threadIdx.x) & 7) * 4)))))[0];
  __syncthreads();
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(0)] * B_shared[(((int)threadIdx.x))]));
  compute_local[(16)] = (compute_local[(16)] + (A_shared[(0)] * B_shared[((((int)threadIdx.x) + 16))]));
  compute_local[(1)] = (compute_local[(1)] + (A_shared[(2)] * B_shared[(((int)threadIdx.x))]));
  compute_local[(17)] = (compute_local[(17)] + (A_shared[(2)] * B_shared[((((int)threadIdx.x) + 16))]));
  compute_local[(2)] = (compute_local[(2)] + (A_shared[(4)] * B_shared[(((int)threadIdx.x))]));
  compute_local[(18)] = (compute_local[(18)] + (A_shared[(4)] * B_shared[((((int)threadIdx.x) + 16))]));
  compute_local[(3)] = (compute_local[(3)] + (A_shared[(6)] * B_shared[(((int)threadIdx.x))]));
  compute_local[(19)] = (compute_local[(19)] + (A_shared[(6)] * B_shared[((((int)threadIdx.x) + 16))]));
  compute_local[(4)] = (compute_local[(4)] + (A_shared[(8)] * B_shared[(((int)threadIdx.x))]));
  compute_local[(20)] = (compute_local[(20)] + (A_shared[(8)] * B_shared[((((int)threadIdx.x) + 16))]));
  compute_local[(5)] = (compute_local[(5)] + (A_shared[(10)] * B_shared[(((int)threadIdx.x))]));
  compute_local[(21)] = (compute_local[(21)] + (A_shared[(10)] * B_shared[((((int)threadIdx.x) + 16))]));
  compute_local[(6)] = (compute_local[(6)] + (A_shared[(12)] * B_shared[(((int)threadIdx.x))]));
  compute_local[(22)] = (compute_local[(22)] + (A_shared[(12)] * B_shared[((((int)threadIdx.x) + 16))]));
  compute_local[(7)] = (compute_local[(7)] + (A_shared[(14)] * B_shared[(((int)threadIdx.x))]));
  compute_local[(23)] = (compute_local[(23)] + (A_shared[(14)] * B_shared[((((int)threadIdx.x) + 16))]));
  compute_local[(8)] = (compute_local[(8)] + (A_shared[(16)] * B_shared[(((int)threadIdx.x))]));
  compute_local[(24)] = (compute_local[(24)] + (A_shared[(16)] * B_shared[((((int)threadIdx.x) + 16))]));
  compute_local[(9)] = (compute_local[(9)] + (A_shared[(18)] * B_shared[(((int)threadIdx.x))]));
  compute_local[(25)] = (compute_local[(25)] + (A_shared[(18)] * B_shared[((((int)threadIdx.x) + 16))]));
  compute_local[(10)] = (compute_local[(10)] + (A_shared[(20)] * B_shared[(((int)threadIdx.x))]));
  compute_local[(26)] = (compute_local[(26)] + (A_shared[(20)] * B_shared[((((int)threadIdx.x) + 16))]));
  compute_local[(11)] = (compute_local[(11)] + (A_shared[(22)] * B_shared[(((int)threadIdx.x))]));
  compute_local[(27)] = (compute_local[(27)] + (A_shared[(22)] * B_shared[((((int)threadIdx.x) + 16))]));
  compute_local[(12)] = (compute_local[(12)] + (A_shared[(24)] * B_shared[(((int)threadIdx.x))]));
  compute_local[(28)] = (compute_local[(28)] + (A_shared[(24)] * B_shared[((((int)threadIdx.x) + 16))]));
  compute_local[(13)] = (compute_local[(13)] + (A_shared[(26)] * B_shared[(((int)threadIdx.x))]));
  compute_local[(29)] = (compute_local[(29)] + (A_shared[(26)] * B_shared[((((int)threadIdx.x) + 16))]));
  compute_local[(14)] = (compute_local[(14)] + (A_shared[(28)] * B_shared[(((int)threadIdx.x))]));
  compute_local[(30)] = (compute_local[(30)] + (A_shared[(28)] * B_shared[((((int)threadIdx.x) + 16))]));
  compute_local[(15)] = (compute_local[(15)] + (A_shared[(30)] * B_shared[(((int)threadIdx.x))]));
  compute_local[(31)] = (compute_local[(31)] + (A_shared[(30)] * B_shared[((((int)threadIdx.x) + 16))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(1)] * B_shared[((((int)threadIdx.x) + 32))]));
  compute_local[(16)] = (compute_local[(16)] + (A_shared[(1)] * B_shared[((((int)threadIdx.x) + 48))]));
  compute_local[(1)] = (compute_local[(1)] + (A_shared[(3)] * B_shared[((((int)threadIdx.x) + 32))]));
  compute_local[(17)] = (compute_local[(17)] + (A_shared[(3)] * B_shared[((((int)threadIdx.x) + 48))]));
  compute_local[(2)] = (compute_local[(2)] + (A_shared[(5)] * B_shared[((((int)threadIdx.x) + 32))]));
  compute_local[(18)] = (compute_local[(18)] + (A_shared[(5)] * B_shared[((((int)threadIdx.x) + 48))]));
  compute_local[(3)] = (compute_local[(3)] + (A_shared[(7)] * B_shared[((((int)threadIdx.x) + 32))]));
  compute_local[(19)] = (compute_local[(19)] + (A_shared[(7)] * B_shared[((((int)threadIdx.x) + 48))]));
  compute_local[(4)] = (compute_local[(4)] + (A_shared[(9)] * B_shared[((((int)threadIdx.x) + 32))]));
  compute_local[(20)] = (compute_local[(20)] + (A_shared[(9)] * B_shared[((((int)threadIdx.x) + 48))]));
  compute_local[(5)] = (compute_local[(5)] + (A_shared[(11)] * B_shared[((((int)threadIdx.x) + 32))]));
  compute_local[(21)] = (compute_local[(21)] + (A_shared[(11)] * B_shared[((((int)threadIdx.x) + 48))]));
  compute_local[(6)] = (compute_local[(6)] + (A_shared[(13)] * B_shared[((((int)threadIdx.x) + 32))]));
  compute_local[(22)] = (compute_local[(22)] + (A_shared[(13)] * B_shared[((((int)threadIdx.x) + 48))]));
  compute_local[(7)] = (compute_local[(7)] + (A_shared[(15)] * B_shared[((((int)threadIdx.x) + 32))]));
  compute_local[(23)] = (compute_local[(23)] + (A_shared[(15)] * B_shared[((((int)threadIdx.x) + 48))]));
  compute_local[(8)] = (compute_local[(8)] + (A_shared[(17)] * B_shared[((((int)threadIdx.x) + 32))]));
  compute_local[(24)] = (compute_local[(24)] + (A_shared[(17)] * B_shared[((((int)threadIdx.x) + 48))]));
  compute_local[(9)] = (compute_local[(9)] + (A_shared[(19)] * B_shared[((((int)threadIdx.x) + 32))]));
  compute_local[(25)] = (compute_local[(25)] + (A_shared[(19)] * B_shared[((((int)threadIdx.x) + 48))]));
  compute_local[(10)] = (compute_local[(10)] + (A_shared[(21)] * B_shared[((((int)threadIdx.x) + 32))]));
  compute_local[(26)] = (compute_local[(26)] + (A_shared[(21)] * B_shared[((((int)threadIdx.x) + 48))]));
  compute_local[(11)] = (compute_local[(11)] + (A_shared[(23)] * B_shared[((((int)threadIdx.x) + 32))]));
  compute_local[(27)] = (compute_local[(27)] + (A_shared[(23)] * B_shared[((((int)threadIdx.x) + 48))]));
  compute_local[(12)] = (compute_local[(12)] + (A_shared[(25)] * B_shared[((((int)threadIdx.x) + 32))]));
  compute_local[(28)] = (compute_local[(28)] + (A_shared[(25)] * B_shared[((((int)threadIdx.x) + 48))]));
  compute_local[(13)] = (compute_local[(13)] + (A_shared[(27)] * B_shared[((((int)threadIdx.x) + 32))]));
  compute_local[(29)] = (compute_local[(29)] + (A_shared[(27)] * B_shared[((((int)threadIdx.x) + 48))]));
  compute_local[(14)] = (compute_local[(14)] + (A_shared[(29)] * B_shared[((((int)threadIdx.x) + 32))]));
  compute_local[(30)] = (compute_local[(30)] + (A_shared[(29)] * B_shared[((((int)threadIdx.x) + 48))]));
  compute_local[(15)] = (compute_local[(15)] + (A_shared[(31)] * B_shared[((((int)threadIdx.x) + 32))]));
  compute_local[(31)] = (compute_local[(31)] + (A_shared[(31)] * B_shared[((((int)threadIdx.x) + 48))]));
  for (int x_inner = 0; x_inner < 16; ++x_inner) {
    compute[((((((((int)blockIdx.x) >> 5) * 16384) + (x_inner * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + ((int)threadIdx.x)))] = compute_local[(x_inner)];
    compute[(((((((((int)blockIdx.x) >> 5) * 16384) + (x_inner * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + ((int)threadIdx.x)) + 16))] = compute_local[((x_inner + 16))];
  }
}

