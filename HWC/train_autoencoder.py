import deeptrack as dt
import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
from data_generator import DataGenerator
from autoencoder import Autoencoder
from keras.models import load_model

# generator = DataGenerator()
# generator.generate_dataset()

train_data = np.load('data/train_data.npy')
train_labels = np.load('data/train_labels.npy')
val_data = np.load('data/val_data.npy')
val_labels = np.load('data/val_labels.npy')

nEpochs = 50
bn_size = 3

ae = Autoencoder((64, 64, 1), bn_size)

train_losses = np.zeros(nEpochs)
val_losses = np.zeros(nEpochs)

for epoch in range(nEpochs):
    for idx, batch in enumerate(train_data):
        # batch_labels = train_labels[idx]

        h = ae.model.fit(x=batch, y=batch, verbose=0)
        train_pred = ae.model.predict(batch, verbose=0)
        train_loss = ae.loss(y_pred=train_pred, y_true=batch).numpy()

    train_losses[epoch] = train_loss

    val_pred = ae.model.predict(val_data[epoch], verbose=0)
    val_loss = ae.loss(y_pred=val_pred, y_true=val_data[epoch]).numpy()
    val_losses[epoch] = val_loss

    print(f'--- Epoch nr. {epoch+1:02d} --- MAE (training): {train_loss:.4f} --- MAE (validation): {val_loss:.4f} ---')

    if train_loss <= 0.3 and val_loss <= 0.35:
        ae.model.save(f'models/mod_train_mae{train_loss:.4f}_test_mae{val_loss:.4f}_bn{bn_size}.keras')
        ae.encoder.save(f'models/enc_train_mae{train_loss:.4f}_test_mae{val_loss:.4f}_bn{bn_size}.keras')
        ae.decoder.save(f'models/dec_train_mae{train_loss:.4f}_test_mae{val_loss:.4f}_bn{bn_size}.keras')
    gc.collect()

pd.DataFrame(train_losses).to_csv('csv/training_loss.csv')
pd.DataFrame(val_losses).to_csv('csv/validation_loss.csv')

'''loaded_model = load_model('models/train_mae0.8426_test_mae0.1731_bn1.keras')

rand = np.random.randint(0, 100)
predicted_series = loaded_model.predict(np.array(val_data[rand]), verbose=0)

for idx, image in enumerate(predicted_series):
    plt.figure()
    plt.imshow(image[..., 0])
    plt.savefig(f'img/pred{idx}.png')

for idx, image in enumerate(val_data[rand]):
    plt.figure()
    plt.imshow(image[..., 0])
    plt.savefig(f'img/original{idx}.png')'''

'''ae = Autoencoder(images[0].shape)
model = ae.create_model()
model.build()
model.summary()'''
