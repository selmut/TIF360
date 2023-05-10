from keras.layers import Layer, Dense, Dropout, LayerNormalization, Flatten
import tensorflow as tf
import numpy as np


class Time2Vector(Layer):
    def __init__(self, seq_len):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        self.weights_linear = self.add_weight(name='weight_linear', shape=(int(self.seq_len),),
                                              initializer='uniform', trainable=True)
        self.bias_linear = self.add_weight(name='bias_linear', shape=(int(self.seq_len),),
                                           initializer='uniform', trainable=True)
        self.weights_periodic = self.add_weight(name='weight_periodic', shape=(int(self.seq_len),),
                                                initializer='uniform', trainable=True)
        self.bias_periodic = self.add_weight(name='bias_periodic', shape=(int(self.seq_len),),
                                             initializer='uniform', trainable=True)

    def call(self, x):
        x = tf.math.reduce_mean(x[:, :, :], axis=-1)
        time_linear = self.weights_linear * x + self.bias_linear
        time_linear = tf.expand_dims(time_linear, axis=-1)
        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1)
        # print(tf.concat([time_linear, time_periodic], axis=-1).shape)
        return tf.concat([time_linear, time_periodic], axis=-1)


class SingleAttention(Layer):
    def __init__(self, d_k, d_v):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = Dense(self.d_k, input_shape=input_shape, kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')
        self.key = Dense(self.d_k, input_shape=input_shape, kernel_initializer='glorot_uniform',
                         bias_initializer='glorot_uniform')
        self.value = Dense(self.d_v, input_shape=input_shape, kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        q = self.query(inputs[0])
        k = self.key(inputs[1])
        attn_weights = tf.matmul(q, k, transpose_b=True)
        attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        v = self.value(inputs[2])
        attn_out = tf.matmul(attn_weights, v)
        return attn_out


class MultiAttention(Layer):
    def __init__(self, n_heads, dk, dv, embedding_dim):
        super(MultiAttention, self).__init__()
        self.n_heads = n_heads
        self.dk = dk
        self.dv = dv
        self.embedding_dim = embedding_dim
        self.attention_heads = []

    def build(self, input_shape):
        for i in range(self.n_heads):
            self.attention_heads.append(SingleAttention(self.dk, self.dv))
        self.linear = tf.keras.layers.Dense(self.embedding_dim)

    def call(self, x):
        attention = []
        for i in range(self.nr_heads):
            attention.append(self.attention_heads[i]([x, x, x]))
        concat_attention = tf.concat(attention, axis=-1)
        x = self.linear(concat_attention)
        return x


class TransformerEncoder(Layer):
    def __init__(self, n_heads, dk, dv, embedding_len, seq_len, dropout_rate):
        self.n_heads = n_heads
        self.dk = dk
        self.dv = dv
        self.embedding_len = embedding_len
        self.seq_len = seq_len

    def build(self, input_shape):
        self.t2v = Time2Vector(self.seq_len)
        self.multi_attention = MultiAttention(self.n_heads, self.dk, self.dv, self.embedding_len)
        self.dropout = Dropout(self.dropout_rate)
        self.normalization = LayerNormalization(epsilon=1e-6)
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.dense3 = Dense(self.embedding_len, activation='relu')
        self.dense4 = Dense(input_shape[-1])
        self.flatten = Flatten()

    def call(self, inputs):
        t2v = self.t2v(inputs)
        add = tf.concat([inputs, t2v], axis=-1)
        x = tf.concat([inputs, t2v], axis=-1)
        x = self.multi_attention(x)
        x = self.dropout(x)

        x = x + add
        x = self.normalize(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        x = x + add
        x = self.normalize(x)
        x = self.flatten(x)
        x = self.dense4(x)
        return x
