#include "../common/parser.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>

struct Direction {
  int x = 0;
  int y = -1;
};

struct Position {
  int row;
  int col;
  Direction dir;

  Position() = default;
  Position(const Position& other) : row(other.row), col(other.col), dir(other.dir) {}

  bool operator==(const Position& other) const {
      return row == other.row && 
              col == other.col && 
              dir.x == other.dir.x && 
              dir.y == other.dir.y;
  }

  void rotate(){
     dir = Direction{-dir.y, dir.x};
  }

  friend std::ostream& operator<<(std::ostream& os, const Position& pos) {
    return os << "Position(row=" << pos.row 
              << ", col=" << pos.col 
              << ", dir=(" << pos.dir.x << "," << pos.dir.y << "))\n";
  }
};

void walk_and_mark(std::vector<std::vector<int>>& mat, Position pos){
  while(pos.row < mat[0].size() && pos.row >= 0 && pos.col < mat.size() && pos.col >= 0){
    mat[pos.row][pos.col] = 0;

    if (pos.row+pos.dir.y < mat[0].size() && pos.row+pos.dir.y >= 0 && pos.col+pos.dir.x < mat.size() && pos.col+pos.dir.x >= 0){
      if (mat[pos.row+pos.dir.y][pos.col+pos.dir.x] == 35){
        pos.rotate();
      } 
    }
    pos.row += pos.dir.y;
    pos.col += pos.dir.x;
  }
}

int find_loops(std::vector<std::vector<int>>& init_mat, Position start_pos) {
    int counter = 0;
    Position pos = start_pos;
    
    while(pos.row < init_mat.size() && pos.row >= 0 && 
          pos.col < init_mat[0].size() && pos.col >= 0) {
        
        auto mat = init_mat;
        int wall_row = pos.row + pos.dir.y;
        int wall_col = pos.col + pos.dir.x;
        
        if (wall_row >= 0 && wall_row < mat.size() && 
            wall_col >= 0 && wall_col < mat[0].size()) {
            
            mat[wall_row][wall_col] = 35;  // Place test wall
            
            Position test_pos = start_pos;
            int steps = 0;
            
            while(test_pos.row < mat.size() && test_pos.row >= 0 && 
                  test_pos.col < mat[0].size() && test_pos.col >= 0) {
                
                if (test_pos == start_pos && steps > 0) {
                    counter++;
                    break;
                }
                
                int next_row = test_pos.row + test_pos.dir.y;
                int next_col = test_pos.col + test_pos.dir.x;
                if (next_row >= 0 && next_row < mat.size() && 
                    next_col >= 0 && next_col < mat[0].size() &&
                    mat[next_row][next_col] == 35) {
                    test_pos.rotate();
                }
                
                test_pos.row += test_pos.dir.y;
                test_pos.col += test_pos.dir.x;
                steps++;
            }
        }
        
        int next_row = pos.row + pos.dir.y;
        int next_col = pos.col + pos.dir.x;
        if (next_row >= 0 && next_row < mat.size() && 
            next_col >= 0 && next_col < mat[0].size() &&
            init_mat[next_row][next_col] == 35) {
            pos.rotate();
        }
        pos.row += pos.dir.y;
        pos.col += pos.dir.x;
    }
    
    return counter;
}

int main(int argc, char *argv[]) {
  try {
    namespace fs = std::filesystem;
    fs::path input = "sample.txt";

    auto buffer = Parser::read_file(input);
    auto vec = Parser::parse_input_to_int(buffer, '\n');
    //46 = .  35 = #  94 = ^ 
    
    auto start = Position{};
    for (int i=0; i<vec.size();++i){
      auto ptr = std::find(vec[i].begin(), vec[i].end(), 94);
      if (ptr != vec[i].end()){
          start.row = i;
          start.col = ptr-vec[i].begin();
      }
    }
   
    walk_and_mark(vec, start);

    int total_zeros = std::accumulate(vec.begin(), vec.end(), 0,
    [](int sum, const std::vector<int>& row) {
        return sum + std::count(row.begin(), row.end(), 0);
    });
    
    std::cout << total_zeros << "\n";


    std::cout << find_loops(vec, start);

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
