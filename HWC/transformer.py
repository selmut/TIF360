from keras.layers import LayerNormalization, MultiHeadAttention, Dropout, Dense, Input, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from custom_layers import Time2Vector, SingleAttention
import tensorflow as tf


class Transformer:
    def __init__(self, input_shape, dk, dv, dense_dim, dropout):
        self.input_shape = input_shape
        self.dk = dk
        self.dv = dv
        self.dense_dim = dense_dim
        self.dropout = dropout

        self.optimizer = Adam(learning_rate=0.0001)
        self.model = self.create_transformer()
        self.model.compile(optimizer=self.optimizer, loss='mae')

    def create_transformer(self):
        inputs = Input(shape=self.input_shape)

        t2v = Time2Vector(10)(inputs)
        x = tf.concat([inputs, t2v], axis=-1)
        x = SingleAttention(d_k=self.dk, d_v=self.dv)([x, x, x])

        # x = LayerNormalization(epsilon=1e-6)(inputs)
        # x = MultiHeadAttention(key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout)(x, x)
        x = Dropout(self.dropout)(x)
        res = x + inputs

        x = LayerNormalization(epsilon=1e-6)(res)
        x = Dense(self.dense_dim, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        x = Dense(inputs.shape[-1], activation='relu')(x)

        outputs = x + inputs
        outputs = LayerNormalization(epsilon=1e-6)(outputs)

        model = Model(inputs, outputs)

        return model

