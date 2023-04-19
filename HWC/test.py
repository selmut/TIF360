from keras.losses import MeanAbsoluteError
from keras.models import load_model, Sequential
from keras.optimizers import Adam
import tensorflow as tf
from transformer import Transformer
from custom_layers import Time2Vector
from custom_layers import SingleAttention
import numpy as np

val_data = np.load('data/val_data.npy')
encoder = load_model('models/enc_train_mae0.2423_test_mae0.2677_bn3.keras')
decoder = load_model('models/dec_train_mae0.2423_test_mae0.2677_bn3.keras')

predicted_series = encoder.predict(val_data[0], verbose=0)
bn = predicted_series.shape[-1]  # bottleneck

input_series = predicted_series.reshape(-1, *predicted_series.shape)
'''t2v = Time2Vector(10)(input_series)

concat = tf.concat([input_series, t2v], axis=-1)

transformed = SingleAttention(d_k=9, d_v=9)([concat, concat, concat])'''


transformer = Transformer((10, 3), 9, bn, 32, 0.3)
test = transformer.model(input_series).numpy()[0]

ttest = decoder(test)

