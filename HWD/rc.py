import numpy as np
import pandas as pd


class CTS:

    def __init__(self):
        self.nVis = 3
        self.mOut = 3
        self.xRes = 500
        self.nPred = 500
        self.k = 0.01

    @staticmethod
    def read_input():
        train_frame = pd.read_csv('csv/training-set.csv', header=None)
        test_frame = pd.read_csv('csv/test-set-1.csv', header=None)

        train_data = train_frame.to_numpy()
        test_data = test_frame.to_numpy()

        return train_data, test_data

    def input_weights(self):
        return np.random.normal(0, np.sqrt(0.002), size=(self.xRes, self.nVis))

    def reservoir_weights(self):
        return np.random.normal(0, np.sqrt(2/500), size=(self.xRes, self.xRes))

    @staticmethod
    def update_states(input_pattern, previous_state, input_weights, reservoir_weights):
        return np.tanh(np.dot(reservoir_weights, previous_state) + np.dot(input_weights, input_pattern))

    def ridge(self, xTrain, r):
        rT = np.transpose(r)
        ridge = self.k * np.ones(np.shape(np.matmul(r, rT)))
        return np.matmul(xTrain, np.matmul(rT, np.linalg.inv(np.add(np.matmul(r, rT), ridge))))

    @staticmethod
    def linear(xTrain, r):
        return np.matmul(xTrain, np.transpose(r))

    @staticmethod
    def compute_output(output_weights, previous_state):
        return np.dot(output_weights, previous_state)

    @staticmethod
    def predict_states(output_pattern, previous_state, input_weights, reservoir_weights):
        return np.tanh(np.dot(reservoir_weights, previous_state) + np.dot(input_weights, output_pattern))

    def train(self, xTrain):
        xRes = self.xRes
        tTrain = np.shape(xTrain)[1]

        # init constant weights
        input_weights = self.input_weights()
        reservoir_weights = self.reservoir_weights()

        # initial states set to zero
        r = np.zeros(xRes)

        # reservoir states, init as empty
        reservoir_states = np.zeros((xRes, tTrain))

        for t in range(tTrain):
            x = xTrain[:, t]
            reservoir_states[:, t] = r
            r = self.update_states(x, r, input_weights, reservoir_weights)

        output_weights = self.ridge(xTrain, reservoir_states)

        return input_weights, reservoir_weights, output_weights

    def predict(self, xTest, input_weights, reservoir_weights, output_weights):
        xRes = self.xRes
        nPred = self.nPred
        mOut = self.mOut
        tTest = np.shape(xTest)[1]

        # initial states set to zero
        r = np.zeros(xRes)

        # reservoir states, init as empty
        reservoir_states = np.zeros((xRes, tTest))

        for t in range(tTest-1):
            x = xTest[:, t + 1]
            reservoir_states[:, t] = r
            r = self.update_states(x, r, input_weights, reservoir_weights)

        # compute last output of test data
        output = self.compute_output(output_weights, r)

        predictions = np.zeros((mOut, nPred))
        previous_state = np.copy(r)

        for n in range(nPred):
            current_state = self.predict_states(output, previous_state, input_weights, reservoir_weights)
            output = self.compute_output(output_weights, current_state)
            previous_state = np.copy(current_state)

            predictions[:, n] = output

        print(predictions[1])

        return predictions[0], predictions[1], predictions[2]

    def run(self):
        # read input data
        train_data, test_data = self.read_input()

        # train reservoir computer
        input_weights, reservoir_weights, output_weights = self.train(train_data)

        x, y, z = self.predict(test_data, input_weights, reservoir_weights, output_weights)

        return x, y, z
