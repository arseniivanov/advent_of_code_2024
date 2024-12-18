def read_numbers(line):
    return [int(x) for x in line.strip()]

with open('sample.txt') as f:
    line = read_numbers(f.readline())

line_proc = []
cnt = 0
for c, entry in enumerate(line):
    if c % 2 == 0:
        line_proc.append((str(cnt), entry))
        cnt += 1
    else:
        line_proc.append(('.', entry))

positions = []
current_pos = 0

ptr_start = 0
ptr_end = len(line_proc) - 1

while ptr_start < ptr_end:
    if ptr_start % 2 == 0:
        key, val = line_proc[ptr_start]
        for _ in range(val):
            positions.append((key, current_pos))
            current_pos += 1
        ptr_start += 1
    else:
        ks, vs = line_proc[ptr_start]
        ke, ve = line_proc[ptr_end]
        diff = vs - ve
        if diff < 0:
            search_ptr = ptr_start + 2 
            found_space = False
            while search_ptr < ptr_end:
                k_search, v_search = line_proc[search_ptr]
                if v_search >= ve:
                    for _ in range(ve):
                        positions.append((ke, current_pos))
                        current_pos += 1
                    line_proc[search_ptr] = (k_search, v_search - ve)
                    found_space = True
                    ptr_start += 1
                    break

                search_ptr += 2

            if not found_space:
                ptr_end -= 2
                continue
        else:
            for _ in range(ve):
                positions.append((ke, current_pos))
                current_pos += 1
            line_proc[ptr_start] = (ks, diff)
            ptr_end -= 2

if ptr_start == ptr_end:
    key, val = line_proc[ptr_end]
    if key != '.':
        for _ in range(val):
            positions.append((key, current_pos))
            current_pos += 1
print(positions)
result = sum(int(key) * pos for key, pos in positions if key != '.')
print(result)
