import numpy as np
from collections import defaultdict
from itertools import permutations

def read_file_to_list(filename):
    with open('input.txt', 'r') as file:
        inp = False
        i = []
        j = []
        for line in file.readlines():
            x = line.strip()
            if x == '':
                inp = True
                continue
            if inp:
                j.append(line.strip())
            else:
                i.append(line.strip())
        return i, j

def check_sequence(sequence, rules_dict):
    sequence_set = set(sequence)
    
    for i, current in enumerate(sequence):
        must_come_after = rules_dict[current]
        remaining_numbers = set(sequence[i+1:])
        
        for after_num in must_come_after:
            if after_num in sequence_set:
                if after_num not in remaining_numbers:
                    return False
                
    return True

def find_violation(sequence, pos, rules_dict):
    curr_val = sequence[pos]
    if curr_val not in rules_dict:
        return -1
    for prev_pos in range(pos):
        if sequence[prev_pos] in rules_dict[curr_val]:
            return prev_pos
    return -1

def reorder_and_calc(sequence, rules_dict):
    def fix_sequence(seq):
        for i in range(len(seq)):
            violation_pos = find_violation(seq, i, rules_dict)
            if violation_pos != -1:
                new_seq = (seq[:violation_pos] + 
                          [seq[i]] + 
                          seq[violation_pos:i] + 
                          seq[i+1:])
                return fix_sequence(new_seq)
        return seq

    if check_sequence(sequence, rules_dict):
        return sequence[len(sequence)//2]

    fixed_sequence = fix_sequence(sequence.copy())
    return fixed_sequence[len(fixed_sequence)//2]

if __name__ == "__main__":
    rules, sequences = read_file_to_list('input.txt')
    
    rules_dict = defaultdict(list)
    for rule in rules:
        before, after = map(int, rule.split('|'))
        rules_dict[before].append(after)
    
    total = 0
    total_faulty = 0
    for seq in sequences:
        nums = list(map(int, seq.split(',')))
        if check_sequence(nums, rules_dict):
            middle_idx = len(nums) // 2
            total += nums[middle_idx]
        else:
            total_faulty += reorder_and_calc(nums, rules_dict)

    
    print(f"Sum of middle numbers in valid sequences: {total}")
    print(f"Sum of middle numbers in fault sequences: {total_faulty}")
