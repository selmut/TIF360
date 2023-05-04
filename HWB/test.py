import numpy as np
# from keras.layers import Concatenate
from itertools import *
import torch
from torch.nn import Flatten


mat1 = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print(mat1.flatten())
