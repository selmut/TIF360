import pandas as pd
import numpy as np
from ReservoirComputerClass import ReservoirComputer


train_data_all = pd.read_csv('csv/training-set.csv', header=None).to_numpy()
test_data_all = pd.read_csv('csv/test-set.csv', header=None).to_numpy()


train_data_x2 = np.zeros((3, len(train_data_all[1, :])))
test_data_x2 = np.zeros((3, len(test_data_all[1, :])))

train_data_x2[1, :] = train_data_all[1, :]
test_data_x2[1, :] = test_data_all[1, :]

res = ReservoirComputer(train_data_x2, test_data_x2, [3, 500, 3], 0.1, 10)
loss = res.run()


