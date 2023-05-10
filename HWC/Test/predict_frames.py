from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np


train_data = np.load('data/encoded/train_data.npy')
train_labels = np.load('data/encoded/train_labels.npy')
val_data = np.load('data/encoded/val_data.npy')
val_labels = np.load('data/encoded/val_labels.npy')

loaded_model = load_model(f'models/transformer_nets/model_dk256_dv256_bn4')
encoder = load_model('models/enc_bn4')
decoder = load_model('models/dec_bn4')

n_preds = 30
rand = np.random.randint(0, len(val_data))

tmp = np.copy(val_data)

for n in range(n_preds):
    print(f'Prediction {n+1}/{n_preds}')
    predicted_images = loaded_model.predict(tmp, verbose=0)
    new_val_data = encoder.predict(predicted_images, verbose=0)

    new_val_data = np.reshape(new_val_data, (-1, 1, 1, 4))

    empty = np.zeros(val_data.shape)
    empty[:, :, :8, :] = tmp[:, :, 1:, :]
    empty[:, :, -1:, :] = new_val_data[:, :, :, :]

    new_images = loaded_model.predict(empty, verbose=0)
    tmp = np.copy(empty)

    plt.figure()
    plt.imshow(new_images[rand])
    plt.savefig(f'img/predicted_frames/prediction{n}.png')
    plt.close()

    plt.figure()
    plt.imshow(new_images[rand])
    plt.savefig(f'img/predicted_frames/prediction{n}.png')
    plt.close()



