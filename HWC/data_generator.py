import deeptrack as dt
import numpy as np
import os
import matplotlib.pyplot as plt


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

        #np.save('data/train_data.npy', train_data)
        np.save('data/val_data.npy', val_data)
        #np.save('data/train_labels.npy', train_labels)
        np.save('data/val_labels.npy', val_labels)

    def generate_transformer_dataset(self):
        os.system('rm -rf data/frame_prediction/*')
        train_set, val_set = self.generate_data()

        train_data = []
        train_labels = []
        val_data = []
        val_labels = []

        for batch in train_set:
            batch = np.array(batch)
            train_data.append(batch[:-1, :, :, :])
            train_labels.append(batch[-1:, :, :, :])
        for batch in val_set:
            batch = np.array(batch)
            val_data.append(batch[:-1, :, :, :])
            val_labels.append(batch[-1:, :, :, :])

        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        val_data = np.array(val_data)
        val_labels = np.array(val_labels)

        np.save('data/frame_prediction/train_data.npy', train_data)
        np.save('data/frame_prediction/train_labels.npy', train_labels)
        np.save('data/frame_prediction/val_data.npy', val_data)
        np.save('data/frame_prediction/val_labels.npy', val_labels)

