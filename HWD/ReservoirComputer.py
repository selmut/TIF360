import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ReservoirComputer:

    def __init__(self, num_in, num_r, num_out, num_pred, ridge_param, max_sing):
        self.num_in = num_in
        self.num_r = num_r
        self.num_out = num_out
        self.num_pred = num_pred
        self.ridge_param = ridge_param
        self.max_sing = max_sing

    @staticmethod
    def read_input():
        train_data = pd.read_csv('csv/training-set.csv', header=None).to_numpy()
        test_data = pd.read_csv('csv/test-set-1.csv', header=None).to_numpy()
        return train_data, test_data

    def input_weights(self):
        # np.random.normal(0, np.sqrt(0.002), size=(self.num_r, self.num_in))
        return np.random.uniform(-0.1, 0.1, size=(self.num_r, self.num_in))

    def reservoir_weights(self):
        # np.random.normal(0, np.sqrt(2/500), size=(self.num_r, self.num_r))
        return np.random.uniform(-1, 1, size=(self.num_r, self.num_r))/self.max_sing

    @staticmethod
    def update_states(input_pattern, previous_state, input_weights, reservoir_weights):
        return np.tanh(np.dot(reservoir_weights, previous_state) + np.dot(input_weights, input_pattern))

    def ridge(self, xTrain, r):
        rT = np.transpose(r)
        ridge = self.ridge_param * np.ones(np.shape(np.matmul(r, rT)))
        return np.matmul(xTrain, np.matmul(rT, np.linalg.inv(np.add(np.matmul(r, rT), ridge))))

    def loss(self, output_weights, true_output, output_pattern, reservoir_states):
        prod1 = np.power(np.abs(true_output-output_pattern*reservoir_states), 2)
        prod2 = self.ridge_param/2*np.trace(np.dot(np.transpose(output_weights), output_weights))
        return prod1+prod2


    @staticmethod
    def compute_output(output_weights, previous_state):
        return np.dot(output_weights, previous_state)

    @staticmethod
    def predict_states(output_pattern, previous_state, input_weights, reservoir_weights):
        return np.tanh(np.dot(reservoir_weights, previous_state) + np.dot(input_weights, output_pattern))

    def train(self, xTrain):
        tTrain = np.shape(xTrain)[1]

        # init constant weights
        input_weights = self.input_weights()
        reservoir_weights = self.reservoir_weights()

        # initial states set to zero
        r = np.zeros(self.num_r)

        # reservoir states, init as empty
        reservoir_states = np.zeros((self.num_r, tTrain))

        for t in range(tTrain):
            x = xTrain[:, t]
            reservoir_states[:, t] = r
            r = self.update_states(x, r, input_weights, reservoir_weights)

        output_weights = self.ridge(xTrain, reservoir_states)

        return input_weights, reservoir_weights, output_weights

    def predict(self, xTest, input_weights, reservoir_weights, output_weights):
        tTest = np.shape(xTest)[1]
        loss = 0

        # initial states set to zero
        r = np.zeros(self.num_r)

        # reservoir states, init as empty
        reservoir_states = np.zeros((self.num_r, tTest))

        for t in range(tTest-1):
            x = xTest[:, t + 1]
            reservoir_states[:, t] = r
            r = self.update_states(x, r, input_weights, reservoir_weights)

        # compute last output of test data
        output = self.compute_output(output_weights, r)
        loss += self.loss(output_weights, true_output, output, reservoir_states)  # TODO fix data for verifying preds.

        predictions = np.zeros((self.num_out, self.num_pred))
        previous_state = np.copy(r)

        for n in range(self.num_pred):
            current_state = self.predict_states(output, previous_state, input_weights, reservoir_weights)
            output = self.compute_output(output_weights, current_state)
            previous_state = np.copy(current_state)

            predictions[:, n] = output

        # print(predictions[1])

        return predictions[0], predictions[1], predictions[2]

    def run(self):
        # read input data
        train_data, test_data = self.read_input()

        # train reservoir computer
        input_weights, reservoir_weights, output_weights = self.train(train_data)

        x1, x2, x3 = self.predict(test_data, input_weights, reservoir_weights, output_weights)

        plt.rcParams["figure.figsize"] = [10, 10]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x1, x2, x3, linewidth=1)
        plt.savefig(f'img/predicted_attractors/singular{self.max_sing}.png')
        plt.close()

        plt.figure()
        plt.plot(np.arange(0, len(x2))*0.02, x2)
        plt.savefig('img/predicted_x2.png')
        plt.close()


singulars = np.linspace(7, 14, num=50)

for s in singulars:
    print(f'Maximal singular value: {s:.3f}')
    rc = ReservoirComputer(3, 500, 3, 500, 0.01, s)
    rc.run()
