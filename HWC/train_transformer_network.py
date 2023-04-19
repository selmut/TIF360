import numpy as np
import pandas as pd
import gc
from keras.models import load_model, save_model
from transformer_network import Network
from transformer import Transformer

nEpochs = 50
bn_size = 3

train_data = np.load('data/train_data.npy')
train_labels = np.load('data/train_labels.npy')
val_data = np.load('data/val_data.npy')
val_labels = np.load('data/val_labels.npy')

encoder = load_model('models/enc_train_mae0.2423_test_mae0.2677_bn3.keras')
decoder = load_model('models/dec_train_mae0.2423_test_mae0.2677_bn3.keras')

encoder.trainable = False
decoder.trainable = False

# transformer = Transformer((10, 3), 9, 3, 32, 0.3)
net = Network((64, 64, 1), encoder, decoder, 3)

train_losses = np.zeros(nEpochs)
val_losses = np.zeros(nEpochs)

for epoch in range(nEpochs):
    for idx, batch in enumerate(train_data):
        # batch_labels = train_labels[idx]
        batch_targets = batch[1:, :, :, :]
        batch = batch[:-1, :, :, :]

        h = net.model.fit(x=batch, y=batch_targets, verbose=0)
        train_pred = net.model.predict(batch, verbose=0)
        train_loss = net.loss(y_pred=train_pred, y_true=batch_targets)

    train_losses[epoch] = train_loss

    current_val_data = val_data[epoch]
    val_batch = current_val_data[:-1, :, :, :]
    val_targets = current_val_data[1:, :, :, :]

    val_pred = net.model.predict(val_batch, verbose=0)
    val_loss = net.loss(y_pred=val_pred, y_true=val_targets).numpy()
    val_losses[epoch] = val_loss

    print(f'--- Epoch nr. {epoch+1:02d} --- MAE (training): {train_loss:.4f} --- MAE (validation): {val_loss:.4f} ---')

    '''if train_loss <= 0.3 and val_loss <= 0.35:
        net.model.save(f'models/mod_train_mae{train_loss:.4f}_test_mae{val_loss:.4f}_bn{bn_size}.keras')
        net.encoder.save(f'models/enc_train_mae{train_loss:.4f}_test_mae{val_loss:.4f}_bn{bn_size}.keras')
        net.decoder.save(f'models/dec_train_mae{train_loss:.4f}_test_mae{val_loss:.4f}_bn{bn_size}.keras')'''
    gc.collect()





