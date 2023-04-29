import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE


def visualize(h, color, filename):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.savefig(filename)


def plot_acc(filename):
    train_acc = pd.read_csv('csv/train_accuracies.csv', header=None).to_numpy().T
    test_acc = pd.read_csv('csv/test_accuracies.csv', header=None).to_numpy().T

    plt.figure()
    plt.plot(train_acc[0], train_acc[1])
    plt.plot(test_acc[0], test_acc[1])

    plt.legend(['Training', 'Testing'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(filename)


plot_acc('img/accuracies.png')

