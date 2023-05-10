import numpy as np
import pandas as pd
from keras.models import load_model, save_model
from networkClass import Network
import matplotlib.pyplot as plt
import tensorflow as tf

nEpochs = 50
bn_size = 4

train_data = np.load('data/encoded/train_data.npy')
train_labels = np.load('data/encoded/train_labels.npy')
val_data = np.load('data/encoded/val_data.npy')
val_labels = np.load('data/encoded/val_labels.npy')


encoder = load_model('models/enc_bn4')
decoder = load_model('models/dec_bn4')

encoder.trainable = False
decoder.trainable = False

dv, dk = 256, 256
net = Network((1, 9, bn_size), decoder, bn_size, 9, dv, dk)

train_labels = np.reshape(train_labels, (-1, 64, 64, 1))
val_labels = np.reshape(val_labels, (-1, 64, 64, 1))

h = net.model.fit(train_data, train_labels, epochs=50, shuffle=True, validation_data=(val_data, val_labels))

net.model.save(f'models/transformer_nets/model_dk{dk}_dv{dv}_bn{bn_size}')

losses = h.history['loss']
val_losses = h.history['val_loss']

np.save('data/losses/losses.npy', losses)
np.save('data/losses/val_losses.npy', val_losses)

loaded_model = load_model(f'models/transformer_nets/model_dk{dk}_dv{dv}_bn{bn_size}')

images = loaded_model.predict(val_data, verbose=0)
rand = np.random.randint(0, len(val_data))
image = images[rand, :, :, :]
val_image = val_labels[rand][:, :, 0]

plt.figure()
plt.imshow(image)
plt.savefig(f'img/predicted.png')

plt.figure()
plt.imshow(val_image)
plt.savefig(f'img/original.png')

