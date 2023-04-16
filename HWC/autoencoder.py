from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, ReLU
from keras.models import Sequential
from keras.optimizers import Adam


class Autoencoder:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.lr = 0.001

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

