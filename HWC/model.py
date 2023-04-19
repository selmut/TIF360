from transformer import Transformer
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, ReLU, UpSampling2D, Reshape, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError


class Autoencoder:
    def __init__(self, input_shape, encoder, decoder, bn):
        self.lr = 0.0001
        self.input_shape = input_shape
        self.bn = bn

        self.input = Input(self.input_shape)
        self.encoder = encoder
        self.decoder = decoder

        self.transformer = Transformer((10, self.bn), 9, self.bn, 32, 0.3)
        self.transformer_model = self.transformer.create_transformer()

        self.model = self.build_model()

        self.optimizer = Adam(learning_rate=self.lr)
        self.model.compile(optimizer=self.optimizer, loss='mae')
        self.loss = MeanAbsoluteError()

    def build_model(self):

        encoder = self.encoder
        decoder = self.decoder
        transformer = self.transformer_model

        encoder.trainable = False
        decoder.trainable = False

        encoder_output = encoder(self.input)
        encoder_output = encoder_output.reshape(-1, *encoder_output.shape)

        transformer_output = transformer(encoder_output).numpy()[0]

        decoder_output = decoder(transformer_output)

        model = Model(self.input, decoder_output)
        return model

