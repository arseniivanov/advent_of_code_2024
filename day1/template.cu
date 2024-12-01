#include "../common/parser.hpp"
// CUDA headers
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>

__global__ void compute_kernel(const int32_t *__restrict__ first,
                               const int32_t *__restrict__ second,
                               int32_t *partial_sums, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int local_diff = 0;
  if (idx < n) {
    local_diff = abs(first[idx] - second[idx]);
  }

  __shared__ int32_t shared_mem[256];
  shared_mem[threadIdx.x] = local_diff;
  __syncthreads();

  // Accumulate
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    partial_sums[blockIdx.x] = shared_mem[0];
  }
}

int main() {
  namespace fs = std::filesystem;
  fs::path input = "input.txt";

  auto buffer = Parser::read_file(input);
  auto vec = Parser::parse_input(buffer, '\n');

  std::vector<int> fst_col;
  std::vector<int> snd_col;

  std::string delim = "   ";
  for (auto line : vec) {
    std::string fst = line.substr(0, line.find(delim));
    std::string snd = line.substr(line.find(delim) + 3, line.size());
    fst_col.push_back(std::stoi(fst));
    snd_col.push_back(std::stoi(snd));
  }
  std::sort(fst_col.begin(), fst_col.end());
  std::sort(snd_col.begin(), snd_col.end());

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

  // Part 2
  auto ptr1 = fst_col.begin();
  auto ptr2 = snd_col.begin();
  sum = 0;
  while (ptr1 != fst_col.end() && ptr2 != snd_col.end()) {
    if (*ptr1 < *ptr2) {
      ptr1++;
    } else if (*ptr1 == *ptr2) {
      sum += *ptr1;
      ptr2++;
    } else {
      ptr2++;
    }
  }

  std::cout << sum << "\n";
}
