import numpy as np
import pandas as pd
import gc
from keras.models import load_model, save_model
from transformer_network import Network
import matplotlib.pyplot as plt
import tensorflow as tf

nEpochs = 50
bn_size = 5

train_data = np.load('data/downsampled/train_data.npy')
train_labels = np.load('data/downsampled/train_labels.npy')
val_data = np.load('data/downsampled/val_data.npy')
val_labels = np.load('data/downsampled/val_labels.npy')


# encoder = load_model('models/encoders/enc_train_mae0.0622_test_mae0.1481_bn4')
decoder = load_model('models/decoders/dec_train_mae0.0324_test_mae0.0331_bn5')

# encoder.trainable = False
decoder.trainable = False

dvs = np.arange(2, 10)*3
dks = np.arange(2, 10)*3

loss_cross = np.zeros((len(dvs), len(dks)))
val_loss_cross = np.zeros((len(dvs), len(dks)))

for i, dv in enumerate(dvs):
    for j, dk in enumerate(dks):
        net = Network((1, 9, 5), decoder, bn_size, 9, dv, dk)
        h = net.model.fit(train_data, train_labels, epochs=50, shuffle=True, validation_data=(val_data, val_labels), verbose=0)

        losses = h.history['loss']
        val_losses = h.history['val_loss']

        min_loss = np.argmin(losses)
        loss_cross[i, j] = losses[min_loss]
        val_loss_cross[i, j] = val_losses[min_loss]

        net.model.save(f'models/transformer_nets/model_dk{dk}_dv{dv}_bn{bn_size}')
        print(f'dv: {dv}; dk: {dk} --- minimum loss: {losses[min_loss]:.4f}; validation loss: {val_losses[min_loss]:.4f}')

    np.save('data/losses/losses.npy', loss_cross)
    np.save('data/losses/val_losses.npy', val_loss_cross)

'''loaded_model = load_model('models/test_model_bn5')

images = loaded_model.predict(val_data, verbose=0)

rand = np.random.randint(0, len(val_data))

image = images[rand, :, :, :]
val_image = val_labels[rand][0,:,:,0]


plt.figure()
plt.imshow(image)
plt.savefig(f'img/predicted.png')

plt.figure()
plt.imshow(val_image)
plt.savefig(f'img/original.png')'''

