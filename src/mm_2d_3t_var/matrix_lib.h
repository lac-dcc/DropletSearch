#ifndef MATRIX_LIB
#define MATRIX_LIB

void mm(
    const float* A,
    const float* B,
    float* C,
    const int rows_A,
    const int c_A_r_B,
    const int cols_B
);

void mm_tile(
    const float* A,
    const float* B,
    float* C,
    const int rows_A,
    const int c_A_r_B,
    const int cols_B,
    const int b_size_A,
    const int b_size_B
);

float* init_matrix(const int rows, const int cols, const float init_value);

void print_matrix(const float* matrix, const int rows, const int cols);

float sum_matrix(const float* matrix, const int rows, const int cols);

void zero_matrix(float* matrix, const int rows, const int cols);

#endif
