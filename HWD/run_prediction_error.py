import pandas as pd
import numpy as np

import plots
from ReservoirComputerClass import ReservoirComputer


train_data_all = pd.read_csv('csv/training-set.csv', header=None).to_numpy()
test_data_all = pd.read_csv('csv/test-set.csv', header=None).to_numpy()

train_data_x2 = np.zeros((3, len(train_data_all[1, :])))
test_data_x2 = np.zeros((3, len(test_data_all[1, :])))

train_data_x2[1, :] = train_data_all[1, :]
test_data_x2[1, :] = test_data_all[1, :]

res_single = ReservoirComputer(train_data_x2, test_data_x2, [3, 500, 3], 0.1,  0.02, 0.05)
loss_single, pred_single, val_single = res_single.run()

res_all = ReservoirComputer(train_data_all, test_data_all, [3, 500, 3], 0.1,  0.02, 0.05)
loss_all, pred_all, val_all = res_all.run()

plots.plot_prediction_error(np.array([pred_all, pred_single]), np.array([val_all, val_single]))

