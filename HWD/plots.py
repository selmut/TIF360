import matplotlib.pyplot as plt
import numpy as np


def plot_predicted_attractor(predictions, max_sing):
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(predictions[0], predictions[1], predictions[2], linewidth=1)
    plt.savefig(f'img/predicted_attractors/singular{max_sing}.png')
    plt.close()


def plot_one_coord(predictions, validations, coord, lambda_max=0.93652738):
    plt.figure()
    plt.plot(np.arange(0, len(predictions)) * 0.02, predictions)
    plt.plot(np.arange(0, len(validations)) * 0.02, validations)

    min_val = np.minimum(np.min(predictions), np.min(validations))*1.2
    max_val = np.maximum(np.max(predictions), np.max(predictions))*1.2

    plt.plot(np.ones(30)*lambda_max, np.linspace(min_val, max_val, num=30), 'k--')
    plt.legend(['Predicted', 'True'])
    plt.axis([0, len(predictions)*0.02, min_val, max_val])
    plt.savefig(f'img/predicted_x{coord}.png')
    plt.close()

