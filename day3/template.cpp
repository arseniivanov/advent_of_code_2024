#include "../common/parser.hpp"
#include <iostream>
#include <regex>
#include <string>

int parse_muls(std::string s) {
  std::regex r_mul("mul\\((\\d+),(\\d+)\\)");
  std::regex r_dont("don't\\(\\)");
  std::regex r_do("do\\(\\)");
  int sum = 0;
  bool active = true;
  std::string::const_iterator searchStart(s.cbegin());
  std::smatch matches_mul;
  std::smatch matches_temp;

  while (searchStart != s.end()) {
    if (active) {
      bool found_mul =
          std::regex_search(searchStart, s.cend(), matches_mul, r_mul);
      bool found_dont =
          std::regex_search(searchStart, s.cend(), matches_temp, r_dont);

      // If no matches found, break
      if (!found_mul && !found_dont)
        break;
      // If only mul found, accumualte and cont
      if (found_mul && !found_dont) {
        int x = std::stoi(matches_mul[1]);
        int y = std::stoi(matches_mul[2]);
        sum += x * y;
        searchStart = matches_mul.suffix().first;
      }
      // If only dont found, we are done
      else if (!found_mul && found_dont) {
        break;
      }
      // Both found - compare positions in original string
      else if (matches_mul.position() < matches_temp.position()) {
        int x = std::stoi(matches_mul[1]);
        int y = std::stoi(matches_mul[2]);
        sum += x * y;
        searchStart = matches_mul.suffix().first;
      } else {
        searchStart = matches_temp.suffix().first;
        active = false;
      }
    } else {
      if (std::regex_search(searchStart, s.cend(), matches_temp, r_do)) {
        searchStart = matches_temp.suffix().first;
        active = true;
      } else {
        break;
      }
    }
  }
  return sum;
}
int main(int argc, char *argv[]) {
  namespace fs = std::filesystem;
  fs::path input = "input.txt";

  auto buffer = Parser::read_file(input);
  auto vec = Parser::parse_input(buffer, '\n');

  std::string combined;
  for (const auto &part : vec) {
    combined += part + "\n";
  }

  // Process everything at once
  int sum = parse_muls(combined);

  std::cout << sum << std::endl;

  return 0;
}
