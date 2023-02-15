#ifndef MATRIX_LIB
#define MATRIX_LIB

void mm_ijk(const float* A, const float* B, float* C, const int rows_A, const int c_A_r_B, const int cols_B);
void mm_ikj(const float* A, const float* B, float* C, const int rows_A, const int c_A_r_B, const int cols_B);
void mm_jik(const float* A, const float* B, float* C, const int rows_A, const int c_A_r_B, const int cols_B);
void mm_jki(const float* A, const float* B, float* C, const int rows_A, const int c_A_r_B, const int cols_B);
void mm_kij(const float* A, const float* B, float* C, const int rows_A, const int c_A_r_B, const int cols_B);
void mm_kji(const float* A, const float* B, float* C, const int rows_A, const int c_A_r_B, const int cols_B);

void mm_tile_ijk(const float* A, const float* B, float* C, const int rows_A, const int c_A_r_B, const int cols_B, const int b_size_A, const int b_size_B);
void mm_tile_ikj(const float* A, const float* B, float* C, const int rows_A, const int c_A_r_B, const int cols_B, const int b_size_A, const int b_size_B);
void mm_tile_jik(const float* A, const float* B, float* C, const int rows_A, const int c_A_r_B, const int cols_B, const int b_size_A, const int b_size_B);
void mm_tile_jki(const float* A, const float* B, float* C, const int rows_A, const int c_A_r_B, const int cols_B, const int b_size_A, const int b_size_B);
void mm_tile_kij(const float* A, const float* B, float* C, const int rows_A, const int c_A_r_B, const int cols_B, const int b_size_A, const int b_size_B);
void mm_tile_kji(const float* A, const float* B, float* C, const int rows_A, const int c_A_r_B, const int cols_B, const int b_size_A, const int b_size_B);

float* init_matrix(const int rows, const int cols, const float init_value);

void print_matrix(const float* matrix, const int rows, const int cols);

float sum_matrix(const float* matrix, const int rows, const int cols);

void zero_matrix(float* matrix, const int rows, const int cols);

#endif
