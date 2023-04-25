import numpy as np
import pandas as pd
import gc
from keras.models import load_model, save_model
from transformer_network import Network
import matplotlib.pyplot as plt
import tensorflow as tf

nEpochs = 50
bn_size = 4

train_data = np.load('data/frame_prediction/train_data.npy')
train_labels = np.load('data/frame_prediction/train_labels.npy')
val_data = np.load('data/frame_prediction/val_data.npy')
val_labels = np.load('data/frame_prediction/val_labels.npy')

encoder = load_model('models/encoders/enc_train_mae0.0622_test_mae0.1481_bn4')
decoder = load_model('models/decoders/dec_train_mae0.0622_test_mae0.1481_bn4')

encoder.trainable = False
decoder.trainable = False

net = Network((64, 64, 1), encoder, decoder, bn_size, 9)

# h = net.model.fit(train_data, train_labels, epochs=50, shuffle=True, validation_data=(val_data, val_labels))

train_losses = np.zeros(nEpochs)
val_losses = np.zeros(nEpochs)

for epoch in range(nEpochs):
    for idx, batch in enumerate(train_data):
        # batch_labels = train_labels[idx]
        batch_targets = batch[-1:, :, :, :]
        batch = batch[:-1, :, :, :]

        batch = tf.stack(batch)
        batch_targets = tf.stack(batch_targets)
        print(batch.shape)
        print(batch_targets.shape)

        h = net.model.fit(batch, batch_targets, verbose=0)
        train_pred = net.model.predict(batch, verbose=0)
        train_loss = net.loss(y_pred=train_pred, y_true=batch_targets)

    train_losses[epoch] = train_loss

    current_val_data = val_data[epoch]
    val_batch = current_val_data[:-1, :, :, :]
    val_targets = current_val_data[1:, :, :, :]

    val_pred = net.model.predict(val_batch, verbose=0)
    val_loss = net.loss(y_pred=val_pred, y_true=val_targets).numpy()
    val_losses[epoch] = val_loss

    # print(f'--- Epoch nr. {epoch+1:02d} --- MSE (training): {train_loss:.4f} --- MSE (validation): {val_loss:.4f} ---')
    print(f'--- Epoch nr. {epoch+1:02d} --- MAE (training): {train_loss:.4f} --- MAE (validation): {val_loss:.4f} ---')

    if train_loss <= 1.1 and val_loss <= 1.15:
        # net.model.save(f'models/transnet_train_mse{train_loss:.4f}_test_mse{val_loss:.4f}_bn{bn_size}')
        net.model.save(f'models/transnet_train_mae{train_loss:.4f}_test_mae{val_loss:.4f}_bn{bn_size}')
    gc.collect()


'''loaded_model = load_model('models/transnet_train_mae1.0384_test_mae0.5796_bn4')
rand = np.random.randint(0, len(val_data))
current_val_data = val_data[rand]
val_batch = current_val_data[:-1, :, :, :]
val_targets = current_val_data[1:, :, :, :]

predicted_series = loaded_model.predict(val_batch, verbose=0)

for idx, image in enumerate(predicted_series):
    plt.figure()
    plt.imshow(image[..., 0])
    plt.savefig(f'img/pred{idx}.png')

for idx, image in enumerate(val_targets):
    plt.figure()
    plt.imshow(image[..., 0])
    plt.savefig(f'img/original{idx}.png')'''



