import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plots


class ReservoirComputer:
    def __init__(self, train_data, test_data, reservoir_dimensions, ridge_param, in_var, res_var):
        self.in_channels = reservoir_dimensions[0]
        self.reservoir_dim = reservoir_dimensions[1]
        self.out_channels = reservoir_dimensions[2]
        self.num_pred = 500
        self.ridge_param = ridge_param
        # self.max_sing = max_sing

        self.input_weights = np.random.normal(0, in_var, size=(self.reservoir_dim, self.in_channels))
        self.reservoir_weights = np.random.normal(0, res_var, size=(self.reservoir_dim, self.reservoir_dim))

        '''self.input_weights = np.random.uniform(-0.1, 0.1, size=(self.reservoir_dim, self.in_channels))
        self.reservoir_weights = np.random.uniform(-1, 1, size=(self.reservoir_dim, self.reservoir_dim)) / self.max_sing'''
        self.output_weights = np.random.uniform(-0.1, 0.1, size=(self.reservoir_dim, self.in_channels))  # to be overwritten

        self.train_data = train_data
        self.test_data = test_data

    def update_reservoir_states(self, input_pattern, previous_state):
        return np.tanh(np.dot(self.reservoir_weights, previous_state) + np.dot(self.input_weights, input_pattern))

    def ridge(self, xTrain, r):
        rT = np.transpose(r)
        ridge = self.ridge_param * np.ones(np.shape(np.matmul(r, rT)))
        return np.matmul(xTrain, np.matmul(rT, np.linalg.inv(np.add(np.matmul(r, rT), ridge))))

    '''def loss(self, true_output, output_pattern, reservoir_states):
        prod1 = np.power(np.abs(true_output-output_pattern*reservoir_states), 2)
        prod2 = self.ridge_param/2*np.trace(np.dot(np.transpose(self.output_weights), self.output_weights))
        return prod1+prod2'''

    def loss(self, prediction, validation, tol=10, lambda_max=0.93652738, dt=0.02):
        diff = np.abs(prediction[1]-validation[1])

        for i, d in enumerate(diff):
            div_idx = i
            if d > tol:
                div_idx = i
                break
        return div_idx*lambda_max*dt

    def compute_output(self, previous_state):
        return np.dot(self.output_weights, previous_state)

    def predict_new_states(self, output_pattern, previous_state):
        return np.tanh(np.dot(self.reservoir_weights, previous_state) + np.dot(self.input_weights, output_pattern))

    def train(self):
        tTrain = np.shape(self.train_data)[-1]

        # initial states set to zero
        r = np.zeros(self.reservoir_dim)

        # reservoir states, init as empty
        reservoir_states = np.zeros((self.reservoir_dim, tTrain))

        for t in range(tTrain):
            x = self.train_data[:, t]
            reservoir_states[:, t] = r
            r = self.update_reservoir_states(x, r)

        self.output_weights = self.ridge(self.train_data, reservoir_states)

    def predict(self):
        split_idx = int(np.shape(self.test_data)[-1]-self.num_pred)

        feed_through_data = self.test_data[:, :split_idx]
        pred_val_data = self.test_data[:, split_idx:]

        feed_through_times = np.shape(feed_through_data)[-1]
        loss = 0

        # initial states set to zero
        r = np.zeros(self.reservoir_dim)

        # reservoir states, init as empty
        reservoir_states = np.zeros((self.reservoir_dim, feed_through_times))

        for t in range(feed_through_times-1):
            x = self.test_data[:, t + 1]
            reservoir_states[:, t] = r
            r = self.update_reservoir_states(x, r)

        # compute last output of test data
        output = self.compute_output(r)

        predictions = np.zeros((self.out_channels, self.num_pred))
        previous_state = np.copy(r)

        for n in range(self.num_pred):
            current_state = self.predict_new_states(output, previous_state)
            output = self.compute_output(current_state)
            previous_state = np.copy(current_state)

            predictions[:, n] = output

        loss = self.loss(predictions, pred_val_data)  # TODO fix data for verifying preds.

        return predictions, pred_val_data, loss

    def run(self):
        # train reservoir computer
        self.train()

        predictions, validations, loss = self.predict()

        try:
            plots.plot_predicted_attractor(predictions, self.max_sing)
            plots.plot_one_coord(predictions[1], validations[1], coord=2)
        except:
            plots.plot_one_coord(predictions[1], validations[1], coord=2)

        return loss, predictions, validations
