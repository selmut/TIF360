import pandas as pd
import numpy as np
from ReservoirComputerClass import ReservoirComputer


train_data_all = pd.read_csv('csv/training-set.csv', header=None).to_numpy()
test_data_all = pd.read_csv('csv/test-set.csv', header=None).to_numpy()

res = ReservoirComputer(train_data_all, test_data_all, [3, 500, 3], 0.1, 10)
loss = res.run()
