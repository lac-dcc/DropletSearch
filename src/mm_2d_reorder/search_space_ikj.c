#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "matrix_lib.h"

int main(int argc, char** argv) {
  const int rows_A = 1000;
  const int c_A_r_B = 800;
  const int cols_B = 700;
  const int block_step = 8;
  const int max_block = 130;

  float* A = init_matrix(rows_A, c_A_r_B, 1.0F);
  float* B = init_matrix(c_A_r_B, cols_B, 1.0F);
  float* C = init_matrix(rows_A, cols_B, 0.0F);
  clock_t start, end;
  double time;
  
  const int qtd = 10;
  double res[qtd];

  for (int b_size_A = 0; b_size_A < max_block; b_size_A += block_step) {
    for (int b_size_B = 0; b_size_B < max_block; b_size_B += block_step) {
      for (int i = 0; i < qtd; ++i) {
        zero_matrix(C, rows_A, cols_B);
        start = clock();
        if (b_size_A * b_size_B) {
          mm_tile_ikj(A, B, C, rows_A, c_A_r_B, cols_B, b_size_A, b_size_B);
        } else {
          mm_ikj(A, B, C, rows_A, c_A_r_B, cols_B);
        }
        end = clock();
        time = ((double) (end - start)) / CLOCKS_PER_SEC;
        res[i] = time;
      }
      double sum = sum_matrix(C, rows_A, cols_B);
      printf("%8.2f, %d, %d", sum, b_size_A, b_size_B);
      for (int i = 0; i < qtd; i++) {
        printf(", %.4lf", res[i]);
      }
      printf("\n");
    }
  }

  free(A);
  free(B);
  free(C);
}
