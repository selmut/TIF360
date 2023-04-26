import deeptrack as dt
import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
from autoencoder import Autoencoder
from keras.models import load_model

train_data = np.load('data/train_data.npy')
train_labels = np.load('data/train_labels.npy')
val_data = np.load('data/val_data.npy')
val_labels = np.load('data/val_labels.npy')

train_data = np.reshape(train_data, (10_000, 64, 64))
val_data = np.reshape(val_data, (10_000, 64, 64))

nEpochs = 50
bn_size = 4

ae = Autoencoder((64, 64, 1), bn_size)

'''h = ae.model.fit(train_data, train_data, epochs=50, shuffle=True, batch_size=10, validation_data=(val_data, val_data))

ae.model.save(f'models/mod_bn{bn_size}')
ae.encoder.save(f'models/enc_bn{bn_size}')
ae.decoder.save(f'models/dec_bn{bn_size}')'''

loaded_model = load_model(f'models/mod_bn{bn_size}')

val_data = np.reshape(val_data, (1000, 10, 64, 64))

rand = np.random.randint(0, 100)
predicted_series = loaded_model.predict(val_data[rand], verbose=0)


for idx, image in enumerate(predicted_series):
    plt.figure()
    plt.imshow(image)
    plt.savefig(f'img/pred{idx}.png')

for idx, image in enumerate(val_data[rand]):
    plt.figure()
    plt.imshow(image)
    plt.savefig(f'img/original{idx}.png')

