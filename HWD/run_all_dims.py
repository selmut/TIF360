import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ReservoirComputerClass import ReservoirComputer

train_data_all = pd.read_csv('csv/training-set.csv', header=None).to_numpy()
test_data_all = pd.read_csv('csv/test-set.csv', header=None).to_numpy()

singulars = np.arange(1, 2000)*0.01
n_reals = 1

losses = np.zeros((n_reals, len(singulars)))

for n in range(n_reals):
    print(f'\nRealisation nr. {n+1}/{n_reals}...\n')
    for i, sing in enumerate(singulars):
        res = ReservoirComputer(train_data_all, test_data_all, [3, 500, 3], 0.1, sing)
        loss = res.run()
        losses[n, i] = loss

        print(f'Current singular value: {sing:.04f}, current loss: {loss:.04f}')

loss = np.mean(losses, axis=0)

plt.figure()
plt.plot(np.log(singulars), np.log(loss))
plt.savefig('img/error_sing.png')

