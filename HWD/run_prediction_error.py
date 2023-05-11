import pandas as pd
import numpy as np

import plots
from ClassReservoirComputer import ReservoirComputer

train_data_all = pd.read_csv('csv/training-set.csv', header=None).to_numpy()
test_data_all = pd.read_csv('csv/test-set.csv', header=None).to_numpy()

train_data_x2 = np.zeros((3, len(train_data_all[1, :])))
test_data_x2 = np.zeros((3, len(test_data_all[1, :])))

train_data_x2[1, :] = train_data_all[1, :]
test_data_x2[1, :] = test_data_all[1, :]


n_reals = 50

preds_all = np.zeros((n_reals, 500))
vals_all = np.zeros((n_reals, 500))

preds_single = np.zeros((n_reals, 500))
vals_single = np.zeros((n_reals, 500))

for n in range(n_reals):
    print(f'Realisation nr. {n + 1}/{n_reals}...')
    res_single = ReservoirComputer(train_data_x2, test_data_x2, [3, 500, 3], 0.1, 0.02, 0.05)
    loss_single, pred_single, val_single = res_single.run()

    res_all = ReservoirComputer(train_data_all, test_data_all, [3, 500, 3], 0.1, 0.02, 0.05)
    loss_all, pred_all, val_all = res_all.run()

    preds_all[n, :] = pred_all[1, :]
    preds_single[n, :] = pred_single[1, :]

    vals_all[n, :] = val_all[1, :]
    vals_single[n, :] = val_single[1, :]

preds_all = np.mean(preds_all, axis=0)
preds_single = np.mean(preds_single, axis=0)
vals_all = np.mean(vals_all, axis=0)
vals_single = np.mean(vals_single, axis=0)

plots.plot_prediction_error(np.array([preds_all, preds_single]), np.array([vals_all, vals_single]))

