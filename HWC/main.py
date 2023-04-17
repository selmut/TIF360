import deeptrack as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_generator import DataGenerator
from autoencoder import Autoencoder

generator = DataGenerator()
train_data, val_data = generator.generate_data()

ae = Autoencoder((64, 64, 1))
model = ae.create_model()

nEpochs = 50

training_loss = np.zeros(nEpochs)
validation_loss = np.zeros(nEpochs)

for epoch in range(nEpochs):
    for idx, batch in enumerate(train_data):
        train_labels = generator.generate_batch_labels(batch)
        h = model.fit(x=np.array(batch), y=train_labels, verbose=0)

        predicted_labels = model.predict(np.array(batch), verbose=0)
        mae_train = np.sum(np.abs(predicted_labels-train_labels))/2/len(train_labels)

    val_labels = generator.generate_batch_labels(val_data[epoch])
    val_img = np.array(val_data[epoch])

    predicted_labels = model.predict(val_img, verbose=0)
    mae_val = np.sum(np.abs(predicted_labels-val_labels))/2/len(val_labels)

    training_loss[epoch] = mae_train
    validation_loss[epoch] = mae_val

    print(f'--- Epoch nr. {epoch + 1} --- MAE (training): {mae_train} --- MAE (test): {mae_val} ---')

pd.DataFrame(training_loss).to_csv('csv/training_loss.csv')
pd.DataFrame(validation_loss).to_csv('csv/validation_loss.csv')


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
