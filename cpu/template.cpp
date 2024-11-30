#include "../common/parser.hpp"
#include <iostream>

int main(int argc, char *argv[]) {
  try {
    namespace fs = std::filesystem;
    fs::path input = "input.txt";

    auto buffer = Parser::read_file(input);
    auto vec = Parser::parse_input(buffer, '\n');

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
