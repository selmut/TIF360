import pandas as pd
from matplotlib import pyplot as plt

PLOT_PATH = 'data/plots/'
REWARDS_PATH = 'data/rewards/'


def plot_rewards(filename):
    plt.figure()
    r_data = pd.read_csv(REWARDS_PATH+'r.csv').to_numpy().T
    plt.plot(r_data[0, :], r_data[1, :])
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.ticklabel_format(axis='both', style='sci', scilimits=[-1, 2])
    # plt.show()
    plt.savefig(PLOT_PATH+filename)


plot_rewards('2a.png')
