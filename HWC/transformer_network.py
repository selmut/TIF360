from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, ReLU, UpSampling2D, Reshape, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError, MeanSquaredError
from custom_layers import Transformer
import tensorflow as tf


class Network:
    def __init__(self, input_shape, encoder, decoder, bn, seq_len):
        self.lr = 0.001
        self.input_shape = input_shape
        self.bn = bn
        self.seq_len = seq_len

        self.input = Input(self.input_shape)
        self.encoder = encoder
        self.decoder = decoder

        self.transformer1 = Transformer(6, 6, 6, 0.4, self.seq_len, 8)
        self.transformer2 = Transformer(6, 6, 6, 0.4, self.seq_len, 8)

        self.model = self.build_model()

        self.optimizer = Adam(learning_rate=self.lr)
        '''self.model.compile(optimizer=self.optimizer, loss='mse')
        self.loss = MeanSquaredError()'''
        self.model.compile(optimizer=self.optimizer, loss='mae')
        self.loss = MeanAbsoluteError()

    def build_model(self):
        self.encoder.trainable = False
        self.decoder.trainable = False

        self.input = tf.reshape(self.input, (-1, 64, 64, 1))
        encoder_output = self.encoder(self.input)
        encoder_output = tf.reshape(encoder_output, (-1, 9, self.bn))

        # encoder_output = tf.reshape(encoder_output, (-1, 10))
        transformer_output = self.transformer1(encoder_output)
        # transformer_output = self.transformer2(transformer_output) # TODO implement double stacked transformers
        # transformer_output = tf.reshape(transformer_output, (-1, self.bn))

        decoder_output = self.decoder(transformer_output)
        # decoder_output = tf.reshape(decoder_output, (-1, 1, self.bn))
        model = Model(self.input, decoder_output)
        return model

