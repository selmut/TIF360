import deeptrack as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_generator import DataGenerator
from autoencoder import Autoencoder

generator = DataGenerator()
train_data, val_data = generator.generate_data()

ae = Autoencoder((64, 64, 1))
model = ae.model

nEpochs = 50

train_losses = np.zeros(nEpochs)
val_losses = np.zeros(nEpochs)

for epoch in range(nEpochs):
    for idx, batch in enumerate(train_data):
        train_labels = generator.generate_batch_labels(batch)
        h = model.fit(x=np.array(batch), y=np.array(batch), verbose=0)
        train_pred = model.predict(np.array(batch), verbose=0)
        train_loss = ae.loss(y_pred=train_pred, y_true=np.array(batch)).numpy()

    train_losses[epoch] = train_loss

    val_pred = model.predict(np.array(val_data[epoch]), verbose=0)
    val_loss = ae.loss(y_pred=val_pred, y_true=np.array(val_data[epoch])).numpy()
    val_losses[epoch] = val_loss

    print(f'--- Epoch nr. {epoch + 1} --- MAE (training): {train_loss} --- MAE (test): {val_loss} ---')

pd.DataFrame(train_losses).to_csv('csv/training_loss.csv')
pd.DataFrame(val_losses).to_csv('csv/validation_loss.csv')


'''for idx, image in batch0:
    plt.figure()
    plt.imshow(image[..., 0])
    plt.savefig(f'img/0{idx}.png')

for idx, image in batch1:
    plt.figure()
    plt.imshow(image[..., 0])
    plt.savefig(f'img/1{idx}.png')'''

'''ae = Autoencoder(images[0].shape)
model = ae.create_model()
model.build()
model.summary()'''
