#include "../common/parser.hpp"
// CUDA headers
#include <cstdint>
#include <cuda_runtime.h>

__global__ void compute_kernel(/* params */) {
  // Kernel code
}

int main() {
  namespace fs = std::filesystem;
  fs::path input = "input.txt";

  auto buffer = Parser::read_file(input);
  auto vec = Parser::parse_input(buffer, '\n');

  char *d_data;
  cudaMalloc(&d_data, vec.size() * sizeof(int32_t));

  cudaMemcpy(d_data, vec.data(), vec.size() * sizeof(int32_t),
             cudaMemcpyHostToDevice);

  // Launch kernel
  const int inp_len = 512;
  dim3 block(32, 8);
  dim3 grid(((inp_len + block.x - 1) / block.x),
            (inp_len + block.y - 1) / block.y);

  compute_kernel<<<grid, block>>>(/* params */);
}
