#include "../common/parser.hpp"
// CUDA headers
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>

__global__ void compute_kernel(const int32_t *__restrict__ first,
                               const int32_t *__restrict__ second,
                               int32_t *partial_sums, int n) {}

std::vector<std::vector<int>>
rotate_90_clockwise(const std::vector<std::vector<int>> &matrix) {
  int n = matrix.size();
  std::vector<std::vector<int>> rotated(n, std::vector<int>(n));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      rotated[j][n - 1 - i] = matrix[i][j];
    }
  }
  return rotated;
}

std::vector<std::vector<int>>
get_diagonal_transform(const std::vector<std::vector<int>> &matrix) {
  int n = matrix.size();
  std::vector<std::vector<int>> diag(n, std::vector<int>(n));
  // TODO
  return diag;
}

std::vector<std::vector<std::vector<int>>>
get_rotations(std::vector<std::vector<int>> mat) {
  std::vector<std::vector<std::vector<int>>> variations;

  auto tempMat = mat;
  for (int i = 0; i < 4; i++) {
    variations.push_back(tempMat);
    variations.push_back(get_diagonal_transform(tempMat));
    tempMat = rotate_90_clockwise(tempMat);
  }

  return variations;
}

int main() {
  namespace fs = std::filesystem;
  fs::path input = "input.txt";

  auto buffer = Parser::read_file(input);
  auto mat = Parser::parse_input_to_int(buffer, '\n');

  auto perms = get_rotations(mat);

  // TOOD
  int32_t *d_first, *d_second;

  cudaMalloc(&d_first, fst_col.size() * sizeof(int32_t));
  cudaMalloc(&d_second, snd_col.size() * sizeof(int32_t));

  cudaMemcpy(d_first, fst_col.data(), fst_col.size() * sizeof(int32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_second, snd_col.data(), snd_col.size() * sizeof(int32_t),
             cudaMemcpyHostToDevice);

  int n = 1000;
  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);

  int32_t *d_partial_sums;
  cudaMalloc(&d_partial_sums, grid.x * sizeof(int32_t));
  compute_kernel<<<grid, block>>>(d_first, d_second, d_partial_sums, n);

  int32_t *h_partial_sums = new int32_t[grid.x];
  cudaMemcpy(h_partial_sums, d_partial_sums, grid.x * sizeof(int32_t),
             cudaMemcpyDeviceToHost);

  int32_t sum = 0;
  for (int i = 0; i < grid.x; i++) {
    sum += h_partial_sums[i];
  }
  std::cout << sum << "\n";

  cudaFree(d_first);
  cudaFree(d_second);
  cudaFree(d_partial_sums);
  delete[] h_partial_sums;
}
