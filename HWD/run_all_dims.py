import pandas as pd
import numpy as np
from ClassReservoirComputer import ReservoirComputer


train_data_all = pd.read_csv('csv/training-set.csv', header=None).to_numpy()
test_data_all = pd.read_csv('csv/test-set.csv', header=None).to_numpy()


res = ReservoirComputer(train_data_all, test_data_all, [3, 500, 3], 0.1, 0.02, 0.05)
loss, pred, val = res.run()

