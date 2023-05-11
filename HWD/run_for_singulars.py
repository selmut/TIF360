import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from ClassReservoirComputer import ReservoirComputer

train_data_all = pd.read_csv('csv/training-set.csv', header=None).to_numpy()
test_data_all = pd.read_csv('csv/test-set.csv', header=None).to_numpy()

train_data_x2 = np.zeros((3, len(train_data_all[1, :])))
test_data_x2 = np.zeros((3, len(test_data_all[1, :])))

train_data_x2[1, :] = train_data_all[1, :]
test_data_x2[1, :] = test_data_all[1, :]

in_var = 0.02
res_vars = np.linspace(1e-2, 1e2, num=50)/500

n_reals = 10
losses = np.zeros((2, n_reals, len(res_vars)))
singulars = np.zeros((2, n_reals, len(res_vars)))

for n in range(n_reals):
    print(f'\nRealisation nr. {n + 1}/{n_reals}...\n')
    for i, res_var in enumerate(res_vars):
        print(f'Current input variance value: {in_var:.06f}, current reservoir variance: {res_var:.06f}')
        res1 = ReservoirComputer(train_data_all, test_data_all, [3, 500, 3], 0.1, in_var, res_var)
        res2 = ReservoirComputer(train_data_x2, test_data_x2, [3, 500, 3], 0.1, in_var, res_var)

        loss1, pred1, val1 = res1.run()
        loss2, pred2, val2 = res2.run()

        losses[0, n, i] = loss1
        losses[1, n, i] = loss2

        u1, s1, vh1 = np.linalg.svd(res1.reservoir_weights)
        u2, s2, vh2 = np.linalg.svd(res2.reservoir_weights)

        singulars[0, n, i] = s1[0]
        singulars[1, n, i] = s2[0]

loss = np.mean(losses, axis=1)
plt.figure()
plt.plot(res_vars, loss[0, :])
plt.plot(res_vars, loss[1, :])
plt.xlabel(r'$\sigma_A^2$')
plt.ylabel('Avg. nr. of predicted Lyapunov times')
plt.legend(['Full information', 'Partial information'])
plt.savefig('img/performance_variance.png')
plt.close()

singular = np.mean(singulars, axis=1)
plt.figure()
plt.plot(singular[0], loss[0, :])
plt.plot(singular[1], loss[1, :])
plt.xlabel(r'Maximum singular value')
plt.ylabel('Avg. nr. of predicted Lyapunov times')
plt.legend(['Full information', 'Partial information'])
plt.savefig('img/performance_singular.png')
plt.close()

# Chosen 0.05 and 0.02 as optimal parameters
'''n_reals = 1000
losses = np.zeros((2, n_reals))
for n in range(n_reals):
    print(f'Realisation nr. {n+1}/{n_reals}')
    res1 = ReservoirComputer(train_data_all, test_data_all, [3, 500, 3], 0.1, 0.02, 0.05)
    res2 = ReservoirComputer(train_data_x2, test_data_x2, [3, 500, 3], 0.1, 0.02, 0.05)

    loss1, pred1, val1 = res1.run()
    loss2, pred2, val2 = res2.run()

    losses[0, n] = loss1
    losses[1, n] = loss2

plt.figure()
plt.hist(losses[0, :], bins=80, color='tab:blue')
plt.xlabel('Nr. of predicted Lyapunov times')
plt.savefig('img/performance/hist_all.png')
plt.show()
plt.close()

plt.figure()
plt.hist(losses[1, :], bins=80, color='tab:orange')
plt.xlabel('Number of predicted Lyapunov times')
plt.savefig('img/performance/hist_x2.png')
plt.show()
plt.close()'''

