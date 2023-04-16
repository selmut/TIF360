import deeptrack as dt
import numpy as np
import matplotlib.pyplot as plt
from data_generator import DataGenerator
from autoencoder import Autoencoder

generator = DataGenerator()
train_data, val_data = generator.generate_data()

ae = Autoencoder((64, 64, 1))
model = ae.create_model()

# training

for idx, batch in enumerate(train_data):
    train_labels = generator.generate_batch_labels(batch)

    rand = np.random.randint(0, 100)
    val_labels = generator.generate_batch_labels(val_data[rand])

    train_img = np.array(batch)
    val_img = np.array(val_data[rand])

    h = model.fit(x=train_img, y=train_labels, validation_data=(val_img, val_labels), epochs=40)
    print(f'Batch nr. {idx+1}')

predicted_labels = model.predict(val_data)
mae = np.sum(np.abs(predicted_labels-val_labels))/2/len(val_labels)
print(f'Mean average error: {mae}')

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
