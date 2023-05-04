import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(self, sigma, r, b, x0, dt, nTimes, nInit):
        self.sigma = sigma
        self.r = r
        self.b = b
        self.x1 = x0[0]
        self.x2 = x0[1]
        self.x3 = x0[2]
        self.t0 = 0
        self.dt = dt
        self.nTimes = nTimes
        self.nInit = nInit

    def dx1(self, x1, x2, x3):
        return -self.sigma*x1+self.sigma*x2

    def dx2(self, x1, x2, x3):
        return -x1*x3+self.r*x1-x2

    def dx3(self, x1, x2, x3):
        return x1*x2-self.b*x3

    def generate_series(self):
        x_sol = np.zeros((3, self.nTimes))
        times = np.zeros(self.nTimes)

        for t in range(self.nInit):
            self.x1 = self.x1 + self.dt*self.dx1(self.x1, self.x2, self.x3)
            self.x2 = self.x2 + self.dt*self.dx2(self.x1, self.x2, self.x3)
            self.x3 = self.x3 + self.dt*self.dx3(self.x1, self.x2, self.x3)

        x_sol[:, 0] = [self.x1, self.x2, self.x3]

        for t in range(self.nTimes-1):
            times[t+1] = times[t]+self.dt

            self.x1 = self.x1 + self.dt*self.dx1(self.x1, self.x2, self.x3)
            self.x2 = self.x2 + self.dt*self.dx2(self.x1, self.x2, self.x3)
            self.x3 = self.x3 + self.dt*self.dx3(self.x1, self.x2, self.x3)

            x_sol[0, t+1] = self.x1
            x_sol[1, t+1] = self.x2
            x_sol[2, t+1] = self.x3

        return times, x_sol

    def plot_save_series(self):
        times, x_sol = self.generate_series()
        split_idx = int(0.8*self.nTimes)

        train_split = x_sol[:, :split_idx]
        test_split = x_sol[:, split_idx:]

        train_times = times[:split_idx]
        test_times = times[split_idx:]

        pd.DataFrame(train_times).to_csv('csv/training-times.csv', index=None, header=None)
        pd.DataFrame(test_times).to_csv('csv/test-times.csv', index=None, header=None)

        pd.DataFrame(train_split).to_csv('csv/training-set.csv', index=None, header=None)
        pd.DataFrame(test_split).to_csv('csv/test-set-1.csv', index=None, header=None)

        plt.rcParams["figure.figsize"] = [10, 10]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_sol[0], x_sol[1], x_sol[2], linewidth=1)
        plt.savefig('img/attractor.png')
        plt.close()

        for n in range(3):
            plt.rcParams["figure.figsize"] = [10, 5]
            plt.figure()
            plt.plot(train_times, train_split[n, :])
            plt.xlabel('t')
            plt.ylabel(rf'$x_{n+1}$')
            plt.savefig(f'img/x{n+1}.png')


gen = DataGenerator(10, 28, 8/3, [1, 1, 1], 0.02, 10_000, 500)
gen.plot_save_series()
