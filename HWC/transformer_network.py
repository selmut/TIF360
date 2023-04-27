from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, ReLU, UpSampling2D, Reshape, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError, MeanSquaredError, CategoricalCrossentropy
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

        self.transformer1 = Transformer(self.dk, self.dv, self.bn+2, 0.2, self.seq_len, 16)
        self.transformer2 = Transformer(self.dk, self.dv, self.bn+2, 0.2, 1, 16)

        self.model = self.build_model()

        self.optimizer = Adam(learning_rate=self.lr)

        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy')
        self.loss = CategoricalCrossentropy()

    def build_model(self):
        # self.encoder.trainable = False
        self.decoder.trainable = False

        transformer_output = self.transformer1(self.input)
        transformer_output = tf.reshape(transformer_output, (-1, 1, self.bn))
        transformer_output = self.transformer2(transformer_output)
        # transformer_output = tf.reshape(transformer_output, (-1, 1, self.bn))
        # transformer_output = self.transformer3(transformer_output)

        decoder_output = self.decoder(transformer_output)
        model = Model(self.input, decoder_output)
        return model

