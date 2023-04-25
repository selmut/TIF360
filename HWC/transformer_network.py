from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, ReLU, UpSampling2D, Reshape, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError, MeanSquaredError
from custom_layers import Transformer
import tensorflow as tf


class Network:
    def __init__(self, input_shape, decoder, bn, seq_len, dv, dk):
        self.lr = 0.001
        self.input_shape = input_shape
        self.bn = bn
        self.seq_len = seq_len
        self.dv = dv
        self.dk = dk

        self.input = Input(self.input_shape)
        self.decoder = decoder

        self.transformer1 = Transformer(self.dk, self.dv, 7, 0.4, self.seq_len, 8)
        self.transformer2 = Transformer(self.dk, self.dv, 7, 0.4, self.seq_len, 8)

        self.model = self.build_model()

        self.optimizer = Adam(learning_rate=self.lr)
        '''self.model.compile(optimizer=self.optimizer, loss='mse')
        self.loss = MeanSquaredError()'''
        self.model.compile(optimizer=self.optimizer, loss='mae')
        self.loss = MeanAbsoluteError()

    def build_model(self):
        # self.encoder.trainable = False
        self.decoder.trainable = False

        transformer_output = self.transformer1(self.input)
        # transformer_output = self.transformer2(transformer_output)  # TODO implement double stacked transformers

        decoder_output = self.decoder(transformer_output)
        # decoder_output = tf.reshape(decoder_output, (-1, 1, self.bn))
        model = Model(self.input, decoder_output)
        return model

