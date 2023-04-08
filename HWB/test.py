import numpy as np
from itertools import *

coordinates = list(permutations([-1, 1], r=2))

m = 4
n = 4

x = [[list(i[x:x+m]) for x in range(0, len(i), m)] for i in product([1, -1], repeat=m*n)]

b = [int(x) for x in list(bin(2*3)[2:])]
print(b)

