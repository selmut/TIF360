import numpy as np
from DataGeneratorClass import DataGenerator

dt = 0.001

x0 = np.array([1, 1, 1])

gen = DataGenerator(10, 28, 8/3, x0, dt, 1_000_000, 50_00)
times, series = gen.generate_series()


def jac(x1, x2, x3, sigma=10, r=28, b=8/3):
    return np.array([[-sigma, sigma, 0], [r-x3, -1, -x1], [x2, x1, -b]])


Q = np.eye(3)
Id = np.eye(3)
lambdas = np.zeros(3)

for n in range(len(times)):
    x1s, x2s, x3s = series[0, n], series[1, n], series[2, n]

    M = Id + jac(x1s, x2s, x3s)*dt
    Q, R = np.linalg.qr(np.matmul(M, Q))

    lambdas[0] += np.log(np.abs(R[0, 0]))
    lambdas[1] += np.log(np.abs(R[1, 1]))
    lambdas[2] += np.log(np.abs(R[2, 2]))

lambdas = lambdas/(len(times)*dt)
print(lambdas)
