import numpy as np

def read_file_to_list(filename):
    try:
        # Open and read the file
        with open('input.txt', 'r') as file:
            # Read lines and strip whitespace
            lines = [line.strip() for line in file.readlines()]
        return lines
    except FileNotFoundError:
        print("Error: input.txt file not found")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}") 
        return None

def check_diagonals(matrix, target):
    rows = len(matrix)
    cols = len(matrix[0])
    target_len = len(target)
    
    directions = {
        "down_right": (1, 1),
        "down_left": (1, -1),
        "up_right": (-1, 1),
        "up_left": (-1, -1)
    }
    
    counter = 0
    for row in range(rows):
        for col in range(cols):
            for direction_name, (row_change, col_change) in directions.items():
                match = True
                
                end_row = row + (target_len - 1) * row_change
                end_col = col + (target_len - 1) * col_change
                
                if (end_row < 0 or end_row >= rows or 
                    end_col < 0 or end_col >= cols):
                    continue
                
                for i in range(target_len):
                    curr_row = row + i * row_change
                    curr_col = col + i * col_change
                    if matrix[curr_row][curr_col] != target[i]:
                        match = False
                        break
                
                if match:
                    counter+=1
    
    return counter

def check_x_mas(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    counter = 0
    for row in range(1, rows-1):
        for col in range(1, cols-1):
            if matrix[row][col] != 'A':
                continue
            top_left = matrix[row-1][col-1]
            top_right = matrix[row-1][col+1]
            bottom_left = matrix[row+1][col-1]
            bottom_right = matrix[row+1][col+1]
            
            if (top_left == 'M' and bottom_right == 'S' and
                top_right == 'M' and bottom_left == 'S'):
                counter += 1
            elif (top_left == 'S' and bottom_right == 'M' and
                  top_right == 'S' and bottom_left == 'M'):
                counter += 1
            elif (top_left == 'M' and bottom_right == 'S' and
                  top_right == 'S' and bottom_left == 'M'):
                counter += 1
            elif (top_left == 'S' and bottom_right == 'M' and
                  top_right == 'M' and bottom_left == 'S'):
                counter += 1
    
    return counter
if __name__ == "__main__":
    data = read_file_to_list('input.txt')
    target = 'XMAS'
    matrix = np.array([list(line) for line in data])
    tot = 0

    tot += check_diagonals(matrix, target)
    for row in matrix:
        tot += ''.join(row).count(target)
        tot += ''.join(row)[::-1].count(target)
    for row in np.transpose(matrix):
        tot += ''.join(row).count(target)
        tot += ''.join(row)[::-1].count(target)
    print(tot)

    print(check_x_mas(matrix))
