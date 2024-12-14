from functools import reduce
from operator import mul, add
from itertools import product

def concat(a, b):
    return int(str(a) + str(b))

f = open('input.txt')
lines = [x.strip().split(': ') for x in f.readlines()]
x = {int(x): list(map(int, y.split(' '))) for x, y in lines}

sum = 0
for k, v in x.items():
    ops = product([add, mul, concat], repeat=len(v)-1)
    for op in ops:
        result = v[0]
        for op, num in zip(op, v[1:]):
            result = op(result, num)
            if result > k:
                break
        if result == k:
            sum += k
            break

print(sum)


