#include "parser.hpp"
#include <fstream>
#include <string_view>

std::vector<std::string> Parser::parse_input(std::vector<char> &buffer,
                                             char delimiter) {
  std::vector<std::string> out;
  size_t start = 0;

  for (size_t i = 0; i < buffer.size(); i++) {
    if (buffer[i] == delimiter) {
      std::string_view view(&buffer[start], i - start);
      out.push_back(std::string(view));
      start = i + 1;
    }
  }

  if (start < buffer.size()) {
    std::string_view view(&buffer[start], buffer.size() - start);
    out.push_back(std::string(view));
  }

  return out;
}

std::vector<std::vector<int>>
Parser::parse_input_to_int(std::vector<char> &buffer, char delimiter) {
  std::vector<std::vector<int>> out;
  size_t start = 0;
  std::vector<int> current_row; // Create a temporary row

  for (size_t i = 0; i < buffer.size(); i++) {
    if (buffer[i] == delimiter) {
      // Process the current line
      std::string_view view(&buffer[start], i - start);
      current_row.clear(); // Clear for new row
      for (int j = 0; j < view.size(); j++) {
        current_row.push_back((int)view[j]);
      }
      out.push_back(current_row); // Add the completed row
      start = i + 1;
    }
  }

  // Handle the last row if there's remaining data
  if (start < buffer.size()) {
    std::string_view view(&buffer[start], buffer.size() - start);
    current_row.clear();
    for (int j = 0; j < view.size(); j++) {
      current_row.push_back((int)view[j]);
    }
    out.push_back(current_row);
  }

  return out;
}

std::vector<char> Parser::read_file(const std::filesystem::path &input_path) {
  std::ifstream file(input_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + input_path.string());
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    throw std::runtime_error("Could not read file: " + input_path.string());
  }

  return buffer;
}
