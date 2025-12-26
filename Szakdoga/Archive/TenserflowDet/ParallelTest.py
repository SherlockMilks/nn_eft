import numpy as np

np.set_printoptions(precision=64)

input="parallel_output.csv"

data = []

with open(input, 'r') as f:
    lines = f.readlines()[1:]
    for line in lines:
        line = line.strip().strip('"')
        nums = list(map(float, line.split(',')))
        data.append(nums)


arr = np.array(data, dtype=np.float64)

col_max = np.max(arr, axis=0)
col_min = np.min(arr, axis=0)
col_diff = col_max - col_min

print("Legkisebb értékek oszloponként:", col_min)
print("Legnagyobb értékek oszloponként:", col_max)
print("Max-min különbség oszloponként:", col_diff)


