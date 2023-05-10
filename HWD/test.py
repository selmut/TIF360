import pandas as pd
import numpy as np
from ClassReservoirComputer import ReservoirComputer

variances = np.linspace(0.04, 0.07, num=100)

singulars = np.zeros(100)
for i, var in enumerate(variances):
    normal = np.random.normal(0, var, size=(500, 500))
    u, s, vh = np.linalg.svd(normal)

    singulars[i] = s[0]

print(singulars)
# the interval of ok variances corresponds approximately to singular values (1.79, 3.11)

