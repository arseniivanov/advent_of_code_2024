#pragma once

#include <filesystem>
#include <string>
#include <vector>

class Parser {
public:
  // Main parsing function
  static std::vector<std::string> parse_input(std::vector<char> &buffer,
                                              char delimiter);

  static std::vector<std::vector<int>>
  parse_input_to_int(std::vector<char> &buffer, char delimiter);
  // Utility function to read file into buffer
  static std::vector<char> read_file(const std::filesystem::path &input_path);
};
