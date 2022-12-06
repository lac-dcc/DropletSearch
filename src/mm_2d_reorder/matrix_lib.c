#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "matrix_lib.h"

#include <math.h>

void mm_ijk(
    const float* A,
    const float* B,
    float* C,
    const int rows_A,
    const int c_A_r_B,
    const int cols_B
) {
  for (int i0 = 0; i0 < rows_A; i0++) {
    for (int j0 = 0; j0 < cols_B; j0++) {
      int ij = i0 * cols_B + j0;
      for (int k0 = 0; k0 < c_A_r_B; k0++) {
        int ik = i0 * c_A_r_B + k0;
        int kj = k0 * cols_B + j0;
        C[ij] += A[ik] * B[kj];
      }
    }
  }
}

void mm_ikj(
    const float* A,
    const float* B,
    float* C,
    const int rows_A,
    const int c_A_r_B,
    const int cols_B
) {
  for (int i0 = 0; i0 < rows_A; i0++) {
    for (int k0 = 0; k0 < c_A_r_B; k0++) {
      int ik = i0 * c_A_r_B + k0;
      for (int j0 = 0; j0 < cols_B; j0++) {
        int ij = i0 * cols_B + j0;
        int kj = k0 * cols_B + j0;
        C[ij] += A[ik] * B[kj];
      }
    }
  }
}

void mm_jik(
    const float* A,
    const float* B,
    float* C,
    const int rows_A,
    const int c_A_r_B,
    const int cols_B
) {
  for (int j0 = 0; j0 < cols_B; j0++) {
    for (int i0 = 0; i0 < rows_A; i0++) {
      int ij = i0 * cols_B + j0;
      for (int k0 = 0; k0 < c_A_r_B; k0++) {
        int ik = i0 * c_A_r_B + k0;
        int kj = k0 * cols_B + j0;
        C[ij] += A[ik] * B[kj];
      }
    }
  }
}

void mm_jki(
    const float* A,
    const float* B,
    float* C,
    const int rows_A,
    const int c_A_r_B,
    const int cols_B
) {
  for (int j0 = 0; j0 < cols_B; j0++) {
    for (int k0 = 0; k0 < c_A_r_B; k0++) {
      int kj = k0 * cols_B + j0;
      for (int i0 = 0; i0 < rows_A; i0++) {
        int ij = i0 * cols_B + j0;
        int ik = i0 * c_A_r_B + k0;
        C[ij] += A[ik] * B[kj];
      }
    }
  }
}

void mm_kij(
    const float* A,
    const float* B,
    float* C,
    const int rows_A,
    const int c_A_r_B,
    const int cols_B
) {
  for (int k0 = 0; k0 < c_A_r_B; k0++) {
  for (int i0 = 0; i0 < rows_A; i0++) {
    int ik = i0 * c_A_r_B + k0;
    for (int j0 = 0; j0 < cols_B; j0++) {
        int ij = i0 * cols_B + j0;
        int kj = k0 * cols_B + j0;
        C[ij] += A[ik] * B[kj];
      }
    }
  }
}

void mm_kji(
    const float* A,
    const float* B,
    float* C,
    const int rows_A,
    const int c_A_r_B,
    const int cols_B
) {
  for (int k0 = 0; k0 < c_A_r_B; k0++) {
    for (int j0 = 0; j0 < cols_B; j0++) {
      int kj = k0 * cols_B + j0;
      for (int i0 = 0; i0 < rows_A; i0++) {
        int ik = i0 * c_A_r_B + k0;
        int ij = i0 * cols_B + j0;
        C[ij] += A[ik] * B[kj];
      }
    }
  }
}

void mm_tile_ijk(
    const float* A,
    const float* B,
    float* C,
    const int rows_A,
    const int c_A_r_B,
    const int cols_B,
    const int b_size_A,
    const int b_size_B
) {
  for (int i0 = 0; i0 < rows_A; i0 += b_size_A) {
    int imax = i0 + b_size_A > rows_A ? rows_A : i0 + b_size_A;
    for (int j0 = 0; j0 < cols_B; j0 += b_size_B) {
      int jmax = j0 + b_size_B > cols_B ? cols_B : j0 + b_size_B;
      for (int k0 = 0; k0 < c_A_r_B; ++k0) {
        for (int j1 = j0; j1 < jmax; ++j1) {
          int kj = k0 * cols_B + j1;
          for (int i1 = i0; i1 < imax; ++i1) {
            int ij = i1 * cols_B + j1;
            int ik = i1 * c_A_r_B + k0;
            C[ij] += A[ik] * B[kj];
          }
        }
      }
    }
  }
}

void mm_tile_ikj(
    const float* A,
    const float* B,
    float* C,
    const int rows_A,
    const int c_A_r_B,
    const int cols_B,
    const int b_size_A,
    const int b_size_B
) {
  for (int i0 = 0; i0 < rows_A; i0 += b_size_A) {
    int imax = i0 + b_size_A > rows_A ? rows_A : i0 + b_size_A;
    for (int k0 = 0; k0 < c_A_r_B; ++k0) {
      for (int j0 = 0; j0 < cols_B; j0 += b_size_B) {
        int jmax = j0 + b_size_B > cols_B ? cols_B : j0 + b_size_B;
        for (int j1 = j0; j1 < jmax; ++j1) {
          int kj = k0 * cols_B + j1;
          for (int i1 = i0; i1 < imax; ++i1) {
            int ij = i1 * cols_B + j1;
            int ik = i1 * c_A_r_B + k0;
            C[ij] += A[ik] * B[kj];
          }
        }
      }
    }
  }
}

void mm_tile_jik(
    const float* A,
    const float* B,
    float* C,
    const int rows_A,
    const int c_A_r_B,
    const int cols_B,
    const int b_size_A,
    const int b_size_B
) {
  for (int j0 = 0; j0 < cols_B; j0 += b_size_B) {
    int jmax = j0 + b_size_B > cols_B ? cols_B : j0 + b_size_B;
    for (int i0 = 0; i0 < rows_A; i0 += b_size_A) {
      int imax = i0 + b_size_A > rows_A ? rows_A : i0 + b_size_A;
      for (int k0 = 0; k0 < c_A_r_B; ++k0) {
        for (int j1 = j0; j1 < jmax; ++j1) {
          int kj = k0 * cols_B + j1;
          for (int i1 = i0; i1 < imax; ++i1) {
            int ij = i1 * cols_B + j1;
            int ik = i1 * c_A_r_B + k0;
            C[ij] += A[ik] * B[kj];
          }
        }
      }
    }
  }
}

void mm_tile_jki(
    const float* A,
    const float* B,
    float* C,
    const int rows_A,
    const int c_A_r_B,
    const int cols_B,
    const int b_size_A,
    const int b_size_B
) {
  for (int j0 = 0; j0 < cols_B; j0 += b_size_B) {
    int jmax = j0 + b_size_B > cols_B ? cols_B : j0 + b_size_B;
    for (int k0 = 0; k0 < c_A_r_B; ++k0) {
      for (int i0 = 0; i0 < rows_A; i0 += b_size_A) {
        int imax = i0 + b_size_A > rows_A ? rows_A : i0 + b_size_A;
        for (int j1 = j0; j1 < jmax; ++j1) {
          int kj = k0 * cols_B + j1;
          for (int i1 = i0; i1 < imax; ++i1) {
            int ij = i1 * cols_B + j1;
            int ik = i1 * c_A_r_B + k0;
            C[ij] += A[ik] * B[kj];
          }
        }
      }
    }
  }
}

void mm_tile_kij(
    const float* A,
    const float* B,
    float* C,
    const int rows_A,
    const int c_A_r_B,
    const int cols_B,
    const int b_size_A,
    const int b_size_B
) {
  for (int k0 = 0; k0 < c_A_r_B; ++k0) {
    for (int i0 = 0; i0 < rows_A; i0 += b_size_A) {
      int imax = i0 + b_size_A > rows_A ? rows_A : i0 + b_size_A;
      for (int j0 = 0; j0 < cols_B; j0 += b_size_B) {
        int jmax = j0 + b_size_B > cols_B ? cols_B : j0 + b_size_B;
        for (int j1 = j0; j1 < jmax; ++j1) {
          int kj = k0 * cols_B + j1;
          for (int i1 = i0; i1 < imax; ++i1) {
            int ij = i1 * cols_B + j1;
            int ik = i1 * c_A_r_B + k0;
            C[ij] += A[ik] * B[kj];
          }
        }
      }
    }
  }
}

void mm_tile_kji(
    const float* A,
    const float* B,
    float* C,
    const int rows_A,
    const int c_A_r_B,
    const int cols_B,
    const int b_size_A,
    const int b_size_B
) {
  for (int k0 = 0; k0 < c_A_r_B; ++k0) {
    for (int j0 = 0; j0 < cols_B; j0 += b_size_B) {
      int jmax = j0 + b_size_B > cols_B ? cols_B : j0 + b_size_B;
      for (int i0 = 0; i0 < rows_A; i0 += b_size_A) {
        int imax = i0 + b_size_A > rows_A ? rows_A : i0 + b_size_A;
        for (int j1 = j0; j1 < jmax; ++j1) {
          int kj = k0 * cols_B + j1;
          for (int i1 = i0; i1 < imax; ++i1) {
            int ij = i1 * cols_B + j1;
            int ik = i1 * c_A_r_B + k0;
            C[ij] += A[ik] * B[kj];
          }
        }
      }
    }
  }
}

float* init_matrix(const int rows, const int cols, const float init_value) {
  const unsigned matrix_size = rows * cols;
  //printf("Allocating %u cells\n", matrix_size);
  float* matrix = (float*) malloc(matrix_size * sizeof(float));
  for (int i = 0; i < matrix_size; i++) {
    matrix[i] = init_value * (float)rand()/(float)RAND_MAX;
  }
  return matrix;
}

void print_matrix(const float* matrix, const int rows, const int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%8.2f", matrix[i * cols + j]);
    }
    printf("\n");
  }
}

void zero_matrix(float* matrix, const int rows, const int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      matrix[i * cols + j] = 0.0F;
    }
  }
}

float sum_matrix(const float* matrix, const int rows, const int cols) {
  float sum = 0.0F;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      sum += matrix[i * cols + j];
    }
  }
  return sum;
}
