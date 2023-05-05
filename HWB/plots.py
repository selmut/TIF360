import pandas as pd
from matplotlib import pyplot as plt

PLOT_PATH = 'data/plots/'
REWARDS_PATH = 'data/rewards/'


def plot_avg(filename):
    r = pd.read_csv(REWARDS_PATH + 'r.csv', header=None, index_col=None).to_numpy().T
    r_avg = pd.read_csv(REWARDS_PATH + 'r_avg.csv', header=None, index_col=None).to_numpy().T

    plt.figure()
    plt.plot(r[0], r[1])
    plt.plot(r_avg[0], r_avg[1])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(PLOT_PATH+filename)


plot_avg('2b.png')

