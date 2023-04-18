from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, ReLU, UpSampling2D, Reshape, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError


class Autoencoder:
    def __init__(self, input_shape, bottleneck_size):
        self.lr = 0.0001
        self.input_shape = input_shape
        self.bottleneck_size = bottleneck_size

        self.input = Input(self.input_shape)
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()
        self.model = Model(inputs=self.input, outputs=self.decoder(self.encoder(self.input)))

        self.optimizer = Adam(learning_rate=self.lr)
        self.model.compile(optimizer=self.optimizer, loss='mae')
        self.loss = MeanAbsoluteError()

    def create_encoder(self):
        encoder = Sequential()
        encoder.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=self.input_shape))
        encoder.add(MaxPool2D(pool_size=(2, 2)))
        encoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        encoder.add(MaxPool2D(pool_size=(2, 2)))
        encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        encoder.add(Flatten())
        encoder.add(Dense(64))
        encoder.add(Dense(64))
        encoder.add(Dense(self.bottleneck_size))
        return encoder

    def create_decoder(self):
        decoder = Sequential()
        decoder.add(Dense(64))
        decoder.add(Dense(64))
        decoder.add(Dense(16*16*64))
        decoder.add(Reshape((16, 16, 64)))
        decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        decoder.add(Conv2D(1, (3, 3), padding='same'))
        return decoder

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=self.input_shape))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(2))

        optimizer = Adam(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss='mae')
        return model

