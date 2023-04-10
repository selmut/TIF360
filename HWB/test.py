import numpy as np
# from keras.layers import Concatenate
from itertools import *

empty = -1*np.ones((6, 4))

board = np.ones((4, 4))
tile = -1*np.ones(4)
tile[2] = 1

empty = -1*np.ones((6, 4))
empty[2:,:] = board
empty[0, :] = tile