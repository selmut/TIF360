import numpy as np
import pandas as pd


class CTS:

    def __init__(self, num_in, num_r, num_out, num_preds, gamma):
        self.num_in = num_in
        self.num_r = num_r
        self.num_out = num_out
        self.num_preds = num_preds
        self.gamma = gamma

        self.input_weights = np.random.uniform(-0.1, 0.1, size=(self.num_r, self.num_in))
        self.reservoir_weights = np.random.normal(0, np.sqrt(3/500), size=(self.num_r, self.num_r))

        self.num_preds = num_preds
        self.gamma = gamma

        self.train_data, self.test_data, self.times = self.read_input()

    def read_input(self):
        times = pd.read_csv('csv/times.csv', header=None).to_numpy()
        x1 = pd.read_csv('csv/x1.csv', header=None).to_numpy()
        x2 = pd.read_csv('csv/x2.csv', header=None).to_numpy()
        x3 = pd.read_csv('csv/x3.csv', header=None).to_numpy()

        split_idx = int(0.8 * len(times))

        train_data = np.array([x1[:split_idx], x2[:split_idx], x3[:split_idx]])
        test_data = np.array([x1[split_idx:], x2[split_idx:], x3[split_idx:]])

        return train_data, test_data, times

    def input_weights(self):
        return np.random.normal(0, np.sqrt(0.002), size=(self.num_r, self.num_in))

    def reservoir_weights(self):
        return np.random.normal(0, np.sqrt(2/500), size=(self.num_r, self.num_r))

    def update_states(self, input_pattern, previous_state):
        sum1 = np.dot(self.reservoir_weights, previous_state).reshape((self.num_r, 1))
        sum2 = np.dot(self.input_weights, input_pattern)
        return np.tanh(sum1 + sum2)

    def ridge(self, xTrain, r):
        rT = np.transpose(r)
        ridge = self.gamma * np.ones(np.shape(np.matmul(r, rT)))

        xTrain = xTrain.reshape((xTrain.shape[:-1]))
        return np.matmul(xTrain, np.matmul(rT, np.linalg.inv(np.add(np.matmul(r, rT), ridge))))

    def compute_output(self, previous_state):
        return np.dot(self.output_weights, previous_state)

    def predict_states(self, output_pattern, previous_state):
        return np.tanh(np.dot(self.reservoir_weights, previous_state) + np.dot(self.input_weights, output_pattern))

    def train(self):
        xRes = self.num_r
        tTrain = np.shape(self.train_data)[1]

        # initial states set to zero
        r = np.zeros((xRes, 1))

        # reservoir states, init as empty
        reservoir_states = np.zeros((xRes, tTrain))

        for t in range(tTrain):
            x = self.train_data[:, t]
            reservoir_states[:, t] = r[:, 0]
            r = self.update_states(x, r)

        self.output_weights = self.ridge(self.train_data, reservoir_states)

    def predict(self):
        tTest = np.shape(self.test_data)[1]

        # initial states set to zero
        r = np.zeros((self.num_r, 1))

        # reservoir states, init as empty
        reservoir_states = np.zeros((self.num_r, tTest))

        for t in range(tTest-1):
            x = self.test_data[:, t + 1]
            reservoir_states[:, t] = r[:, 0]
            r = self.update_states(x, r)

        # compute last output of test data
        output = self.compute_output(r)

        predictions = np.zeros((self.num_out, self.num_preds))
        previous_state = np.copy(r)

        for n in range(self.num_preds):
            current_state = self.predict_states(output, previous_state)
            output = self.compute_output(current_state)
            previous_state = np.copy(current_state)

            predictions[:, n] = output[:, 0]

        return predictions

    def run(self):
        # read input data
        # train reservoir computer
        print('Starting training...\n')
        self.train()
        print('Output weights:')
        print(self.output_weights)

        print('\nStarting to predict...')
        preds = self.predict()
        return preds


cts = CTS(3, 500, 3, 1000, 0.1)

preds = cts.run()
