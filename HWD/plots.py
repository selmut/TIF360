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


def plot_one_coord(predictions, validations, coord, lambda_max=0.93652738, dt=0.02):
    plt.figure()
    plt.plot(np.arange(0, len(predictions)) * dt*lambda_max, predictions)
    plt.plot(np.arange(0, len(validations)) * dt*lambda_max, validations)

    min_val = np.minimum(np.min(predictions), np.min(validations))*1.2
    max_val = np.maximum(np.max(predictions), np.max(validations))*1.2

    plt.plot(np.ones(30)*lambda_max, np.linspace(min_val, max_val, num=30), 'k--')
    plt.legend(['Predicted', 'True'])
    plt.xlabel(r'$\lambda_1$t')
    plt.ylabel(rf'$x_{coord}$')
    plt.axis([0, len(predictions)*dt, min_val, max_val])
    plt.savefig(f'img/predicted_x{coord}.png')
    plt.close()


def plot_prediction_error(predictions, validations, lambda_max=0.93652738, dt=0.02):
    plt.figure()

    error_all = np.square(predictions[0, 1] - validations[0, 1])
    error_single = np.square(predictions[1, 1] - validations[1, 1])

    plt.plot(np.arange(0, len(predictions[0, 0])) * dt * lambda_max, error_all)
    plt.plot(np.arange(0, len(predictions[0, 0])) * dt * lambda_max, error_single)

    min_val = np.minimum(np.min(error_all), np.min(error_single))*1.1
    max_val = np.maximum(np.max(error_all), np.max(error_single))*1.1

    plt.plot(np.ones(30)*lambda_max, np.linspace(min_val, max_val, num=30), 'k--')

    plt.xlabel(r'$\lambda_1t$')
    plt.ylabel(r'MSE in $x_2$-axis')
    plt.axis([0, len(predictions[0, 0])*dt, min_val, max_val])
    plt.legend(['Full information', 'Partial information'])
    plt.savefig('img/error_plots/pred_error.png')
