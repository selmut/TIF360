import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(self, sigma, r, b, x0, dt, nTimes):
        self.sigma = sigma
        self.r = r
        self.b = b
        self.x1 = x0[0]
        self.x2 = x0[1]
        self.x3 = x0[2]
        self.t0 = 0
        self.dt = dt
        self.nTimes = nTimes

    def dx1(self, x1, x2, x3):
        return -self.sigma*x1+self.sigma*x2

    def dx2(self, x1, x2, x3):
        return -x1*x3+self.r*x1-x2

    def dx3(self, x1, x2, x3):
        return x1*x2-self.b*x3

    def generate_series(self):
        x_out = np.zeros((3, self.nTimes))
        times = np.zeros(self.nTimes)

        for t in range(self.nTimes-1):
            times[t+1] = times[t]+self.dt

            self.x1 = self.x1 + self.dt*self.dx1(self.x1, self.x2, self.x3)
            self.x2 = self.x2 + self.dt*self.dx2(self.x1, self.x2, self.x3)
            self.x3 = self.x3 + self.dt*self.dx3(self.x1, self.x2, self.x3)

            x_out[0, t+1] = self.x1
            x_out[1, t+1] = self.x2
            x_out[2, t+1] = self.x3

        return times, x_out

    def plot_save_series(self):
        times, x_sol = self.generate_series()

        pd.DataFrame(times[int(22/self.dt):]).to_csv('csv/times.csv', index=None, header=None)
        pd.DataFrame(x_sol[0, int(22/self.dt):]).to_csv('csv/x1.csv', index=None, header=None)
        pd.DataFrame(x_sol[1, int(22/self.dt):]).to_csv('csv/x2.csv', index=None, header=None)
        pd.DataFrame(x_sol[2, int(22/self.dt):]).to_csv('csv/x3.csv', index=None, header=None)

        plt.rcParams["figure.figsize"] = [10, 10]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_sol[0], x_sol[1], x_sol[2], linewidth=0.5)
        plt.savefig('img/attractor.png')
        plt.close()

        for n in range(3):
            plt.rcParams["figure.figsize"] = [10, 5]
            plt.figure()
            plt.plot(times[int(22/self.dt):int(30/self.dt)], x_sol[n, int(22/self.dt):int(30/self.dt)])
            plt.xlabel('t')
            plt.ylabel(rf'$x_{n+1}$')
            plt.savefig(f'img/x{n+1}.png')


gen = DataGenerator(10, 28, 8/3, [1, 1, 1], 0.001, 500_000)
gen.plot_save_series()
