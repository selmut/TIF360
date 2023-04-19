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

        h = net.model.fit(x=batch, y=batch)

    '''train_losses[epoch] = train_loss

    val_pred = ae.model.predict(val_data[epoch], verbose=0)
    val_loss = ae.loss(y_pred=val_pred, y_true=val_data[epoch]).numpy()
    val_losses[epoch] = val_loss

    print(f'--- Epoch nr. {epoch+1:02d} --- MAE (training): {train_loss:.4f} --- MAE (validation): {val_loss:.4f} ---')

    if train_loss <= 0.3 and val_loss <= 0.35:
        ae.model.save(f'models/mod_train_mae{train_loss:.4f}_test_mae{val_loss:.4f}_bn{bn_size}.keras')
        ae.encoder.save(f'models/enc_train_mae{train_loss:.4f}_test_mae{val_loss:.4f}_bn{bn_size}.keras')
        ae.decoder.save(f'models/dec_train_mae{train_loss:.4f}_test_mae{val_loss:.4f}_bn{bn_size}.keras')'''
    gc.collect()





