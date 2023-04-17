import matplotlib.pyplot as plt
import pandas as pd


def plot_acc(filename):
    train_acc = pd.read_csv('csv/training_loss.csv', header=None).to_numpy().T
    test_acc = pd.read_csv('csv/validation_loss.csv', header=None).to_numpy().T

    plt.figure()
    plt.plot(train_acc[0], train_acc[1])
    plt.plot(test_acc[0], test_acc[1])

    plt.legend(['Training', 'Validation'])
    plt.savefig(filename)


plot_acc('img/accuracies.png')


