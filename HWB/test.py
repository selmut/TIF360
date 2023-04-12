import numpy as np
# from keras.layers import Concatenate
from itertools import *
import torch.nn as nn
import torch

state = -1 * np.ones((5, 4))
state = torch.from_numpy(state.reshape(-1, *state.shape))

print(state)
input = torch.randn(32, 1, 5, 5)
#print(input[0])
# With default parameters
m = nn.Flatten()
output = m(state)
print(output)

output.size()
# With non-default parameters
m = nn.Flatten(0, 2)
output = m(input)
output.size()
