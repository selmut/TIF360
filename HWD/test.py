import numpy as np
from ReservoirComputer import ReservoirComputer
from matplotlib import pyplot as plt
import pandas as pd

cts = CTS()
x, y, z = cts.run()

y_DF = pd.DataFrame(y)
y_DF.to_csv('prediction.csv', index=None, header=None)

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
plt.show()
plt.close()


plt.figure()
plt.plot(0.02*np.arange(len(y)), y)
plt.savefig('img/x2_pred.png')
