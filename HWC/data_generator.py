import deeptrack as dt
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import load_model, save_model


# generates a sequence of 10 images
class DataGenerator:
    def __init__(self):
        self.IMAGE_SIZE = 64
        self.sequence_length = 10  # Number of frames per sequence
        self.MIN_SIZE = 0.5e-6
        self.MAX_SIZE = 1.5e-6
        self.MAX_VEL = 10  # Maximum velocity. The higher, the trickier!
        self.MAX_PARTICLES = 3  # Max number of particles in each sequence. The higher, the trickier!

        self.particle = dt.Sequential(self.generate_particle(), position=self.get_position)
        self.optics = self.generate_optics()

    def generate_particle(self):
        return dt.Sphere(intensity=lambda: 10 + 10 * np.random.rand(),
                         radius=lambda: self.MIN_SIZE + np.random.rand() * (self.MAX_SIZE - self.MIN_SIZE),
                         position=lambda: self.IMAGE_SIZE * np.random.rand(2),
                         vel=lambda: self.MAX_VEL * np.random.rand(2), position_unit="pixel",)

    @staticmethod
    def get_position(previous_value, vel):
        newv = previous_value + vel
        for i in range(2):
            if newv[i] > 63:
                newv[i] = 63 - np.abs(newv[i] - 63)
                vel[i] = -vel[i]
            elif newv[i] < 0:
                newv[i] = np.abs(newv[i])
                vel[i] = -vel[i]
        return newv

    def generate_optics(self):
        return dt.Fluorescence(NA=1, output_region=(0, 0, self.IMAGE_SIZE, self.IMAGE_SIZE), magnification=10,
                               resolution=(1e-6, 1e-6, 1e-6), wavelength=633e-9,)

    def generate_data(self):
        sequential_images = dt.Sequence(
            self.optics(self.particle ** (lambda: 1 + np.random.randint(self.MAX_PARTICLES))),
            sequence_length=self.sequence_length,
        )
        dataset = sequential_images >> dt.FlipUD() >> dt.FlipDiagonal() >> dt.FlipLR()

        train_data = [dataset.update()() for i in range(1000)]  # 1000 sequences, each a random sphere moving for 10 frames
        val_data = [dataset.update()() for i in range(1000)]  # 100 sequences, each a random sphere moving for 10 frames

        return train_data, val_data

    def generate_labels(self, data):
        '''train_labels = [generator.generate_labels(train_data0[i]) for i in range(len(train_data0))]
        val_labels = [generator.generate_labels(val_data0[i]) for i in range(len(val_data0))]'''

        labels = np.zeros((len(data), 10, 2))

        for idx, batch in enumerate(data):
            labels[idx, :, :] = np.array([np.array(image.get_property('position')) for image in batch])

        # position = image.get_property("position")
        return labels

    def generate_dataset(self):
        #os.system('rm -rf data/*')
        train_data, val_data = self.generate_data()
        train_labels = self.generate_labels(train_data)
        val_labels = self.generate_labels(val_data)

        np.save('data/train_data.npy', train_data)
        np.save('data/val_data.npy', val_data)
        np.save('data/train_labels.npy', train_labels)
        np.save('data/val_labels.npy', val_labels)

    def generate_encoded_dataset(self):
        # os.system('rm -rf data/downsampled/*')

        train_data = np.load('data/train_data.npy')
        val_data = np.load('data/val_data.npy')

        encoder = load_model('models/enc_bn4')

        train_data_downsampled = []
        train_labels_downsampled = []
        val_data_downsampled = []
        val_labels_downsampled = []

        for idx, batch in enumerate(train_data):

            batch_inputs = batch[:-1, :, :, :]
            batch_labels = batch[-1, :, :, :]

            pred = encoder.predict(batch_inputs, verbose=0)
            pred = np.reshape(pred, (-1, *pred.shape))
            label = np.reshape(batch_labels, (-1, *batch_labels.shape))

            train_data_downsampled.append(pred)
            train_labels_downsampled.append(label)

        for idx, batch in enumerate(val_data):
            batch_inputs = batch[:-1, :, :, :]
            batch_labels = batch[-1, :, :, :]

            pred = encoder.predict(batch_inputs, verbose=0)
            pred = np.reshape(pred, (-1, *pred.shape))
            label = np.reshape(batch_labels, (-1, *batch_labels.shape))

            val_data_downsampled.append(pred)
            val_labels_downsampled.append(label)

        np.save('data/downsampled/train_data.npy', np.array(train_data_downsampled))
        np.save('data/downsampled/train_labels.npy', np.array(train_labels_downsampled))
        np.save('data/downsampled/val_data.npy', np.array(val_data_downsampled))
        np.save('data/downsampled/val_labels.npy', np.array(val_labels_downsampled))


gen = DataGenerator()
gen.generate_encoded_dataset()
