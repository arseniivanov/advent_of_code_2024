#include "../common/parser.hpp"
// CUDA headers
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>

__global__ void pattern_match_140(const int32_t *matrix, int pitch,
                                  int32_t *pattern, int pattern_length,
                                  int32_t *partial_sums) {
  extern __shared__ int32_t sdata[];
  int tid = threadIdx.x;
  int row = blockIdx.x; // Each block handles one row

  int32_t local_sum = 0;
  // Each thread processes multiple elements in the row
  for (int col = tid; col < 140 - pattern_length + 1; col += blockDim.x) {
    bool match = true;
    for (int p = 0; p < pattern_length && match; p++) {
      if (matrix[row * pitch + col + p] != pattern[p]) {
        match = false;
      }
    }
    local_sum += match ? 1 : 0;
  }

  // Shared memory reduction
  sdata[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0)
    partial_sums[row] = sdata[0];
}

__global__ void pattern_match_280(const int32_t *matrix, int pitch,
                                  int32_t *pattern, int pattern_length,
                                  int32_t *partial_sums) {
  extern __shared__ int32_t sdata[];
  int tid = threadIdx.x;
  int row = blockIdx.x; // Each block handles one row

  int32_t local_sum = 0;
  // Each thread processes multiple elements in the row
  for (int col = tid; col < 280 - pattern_length + 1; col += blockDim.x) {
    bool match = true;
    for (int p = 0; p < pattern_length && match; p++) {
      if (matrix[row * pitch + col + p] != pattern[p]) {
        match = false;
      }
    }
    local_sum += match ? 1 : 0;
  }

  // Shared memory reduction
  sdata[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0)
    partial_sums[row] = sdata[0];
}
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
  std::vector<std::vector<int>> diag(2 * n, std::vector<int>(2 * n, 0));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      diag[i + j + 1][-i + j + n] = matrix[i][j];
    }
  }
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

  std::vector<int32_t> pattern = {88, 77, 65, 83}; // Your pattern here
  int pattern_length = pattern.size();

  int32_t *d_pattern;
  cudaMalloc(&d_pattern, pattern_length * sizeof(int32_t));
  cudaMemcpy(d_pattern, pattern.data(), pattern_length * sizeof(int32_t),
             cudaMemcpyHostToDevice);

  int32_t *d_matrices[8];
  int32_t *d_results[8];
  int pitches[8];

  int32_t mat1_size = perms[0].size() * perms[0].size();
  int32_t mat2_size = perms[1].size() * perms[1].size();

  for (int i = 0; i < 8; i += 2) {
    size_t pitch;
    cudaMallocPitch(&d_matrices[i], &pitch, 140 * sizeof(int32_t), 140);
    pitches[i] = pitch / sizeof(int32_t);
    cudaMalloc(&d_results[i], 140 * sizeof(int32_t)); // One sum per row

    cudaMemcpy2D(d_matrices[i], pitch, perms[i].data(), 140 * sizeof(int32_t),
                 140 * sizeof(int32_t), 140, cudaMemcpyHostToDevice);
  }

  for (int i = 1; i < 8; i += 2) {
    size_t pitch;
    cudaMallocPitch(&d_matrices[i], &pitch, 280 * sizeof(int32_t), 280);
    pitches[i] = pitch / sizeof(int32_t);
    cudaMalloc(&d_results[i], 280 * sizeof(int32_t));

    cudaMemcpy2D(d_matrices[i], pitch, perms[i].data(), 280 * sizeof(int32_t),
                 280 * sizeof(int32_t), 280, cudaMemcpyHostToDevice);
  }

  dim3 block_140(256);
  dim3 grid_140(140); // One block per row

  dim3 block_280(256);
  dim3 grid_280(280);

  // Launch kernels
  for (int i = 0; i < 8; i += 2) {
    pattern_match_140<<<grid_140, block_140, block_140.x * sizeof(int32_t)>>>(
        d_matrices[i], pitches[i], d_pattern, pattern_length, d_results[i]);
  }

  for (int i = 1; i < 8; i += 2) {
    pattern_match_280<<<grid_280, block_280, block_280.x * sizeof(int32_t)>>>(
        d_matrices[i], pitches[i], d_pattern, pattern_length, d_results[i]);
  }

  // Collect results
  std::vector<int32_t> total_matches(8, 0);
  for (int i = 0; i < 4; i++) {
    std::vector<int32_t> row_matches(140);
    cudaMemcpy(row_matches.data(), d_results[i], 140 * sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    total_matches[i] =
        std::accumulate(row_matches.begin(), row_matches.end(), 0);
  }

  for (int i = 4; i < 8; i++) {
    std::vector<int32_t> row_matches(280);
    cudaMemcpy(row_matches.data(), d_results[i], 280 * sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    total_matches[i] =
        std::accumulate(row_matches.begin(), row_matches.end(), 0);
  }

  // Cleanup
  cudaFree(d_pattern);
  for (int i = 0; i < 8; i++) {
    cudaFree(d_matrices[i]);
    cudaFree(d_results[i]);
  }

  // Print results
  int sum = 0;
  for (int i = 0; i < 8; i++) {
    sum += total_matches[i];
  }
  std::cout << "Matches: " << sum << "\n";
}
