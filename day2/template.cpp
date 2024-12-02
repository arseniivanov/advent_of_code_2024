#include "../common/parser.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::vector<int> split_to_ints(const std::string &input, char delimiter = ' ') {
  std::vector<int> result;
  std::stringstream ss(input);
  std::string item;

  while (std::getline(ss, item, delimiter)) {
    if (!item.empty()) {
      result.push_back(std::stoi(item));
    }
  }

  return result;
}

int main() {
  namespace fs = std::filesystem;
  fs::path input = "input.txt";

  auto buffer = Parser::read_file(input);
  auto vec = Parser::parse_input(buffer, '\n');

  std::vector<std::vector<int>> entries;
  int sum_good = 0;

#pragma openmp
  for (auto line : vec) {
    auto nums = split_to_ints(line);
    int a = nums[0];
    int b = nums[1];
    int absdiff = abs(a - b);
    if (absdiff > 3 | absdiff == 0) {
      break;
    }
    int diff = b - a;
    for (int i = 2; i < nums.size() - 1; i++) {
      a = nums[i];
      b = nums[i + 1];
      int absdiff = abs(a - b);
      if (absdiff > 3 | absdiff == 0) {
        break;
      }
      if
    }
  }

  std::cout << delim << "\n";
}
