import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_acc(filename):
    train_acc = pd.read_csv('results/bn4/training_loss.csv', header=None).to_numpy().T
    test_acc = pd.read_csv('results/bn4/validation_loss.csv', header=None).to_numpy().T

    plt.figure()
    plt.plot(train_acc[0], train_acc[1])
    plt.plot(test_acc[0], test_acc[1])

    plt.legend(['Training', 'Validation'])
    plt.savefig(filename)


def plot_losses(filename):
    train_acc = np.load('data/losses/losses.npy')
    val_acc = np.load('data/losses/val_losses.npy')

    plt.figure()
    plt.plot(np.arange(0, 100), train_acc)
    plt.plot(np.arange(0, 100), val_acc)

    plt.legend(['Training', 'Validation'])
    plt.savefig(filename)


def plot_loss_maps(data, filename):
    plt.figure()
    plt.imshow(data)
    plt.savefig(filename)


plot_losses('img/transformer_losses.png')
#plot_loss_maps(np.load('data/losses/losses.npy'), 'img/loss_map.png')
#plot_loss_maps(np.load('data/losses/val_losses.npy'), 'img/loss_map.png')
