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
    def __init__(self, nr_heads, d_k, d_v, embedding_dim):
        super(MultiAttention, self).__init__()
        self.nr_heads = nr_heads
        self.d_k = d_k
        self.d_v = d_v
        self.embedding_dim = embedding_dim
        self.attention_heads = []

    def build(self, input_shape):
        for i in range(self.nr_heads):
            self.attention_heads.append(SingleAttention(self.d_k, self.d_v))
        self.linear = tf.keras.layers.Dense(self.embedding_dim)

    def call(self, x):
        attention = []
        for i in range(self.nr_heads):
            attention.append(self.attention_heads[i]([x, x, x]))
        concat_attention = tf.concat(attention, axis=-1)
        x = self.linear(concat_attention)
        return x


class Transformer(Layer):
    def __init__(self, dk, dv, embedding_len, dropout_rate, seq_len, n_heads):
        super(Transformer, self).__init__()
        self.dk = dk
        self.dv = dv
        self.dropout_rate = dropout_rate
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.embedding_len = embedding_len

    def build(self, input_shape):
        self.t2v = Time2Vector(self.seq_len)
        self.multi_attention = MultiAttention(self.n_heads, self.dk, self.dv, 6)
        self.dropout = Dropout(self.dropout_rate)
        self.normalize = LayerNormalization(epsilon=1e-3)
        self.dense1 = Dense(32, activation='relu')
        self.dense2 = Dense(16, activation='relu')
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
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dense3(x)

        x = x + add
        x = self.normalize(x)
        #print(x.shape)
        x = self.flatten(x)
        x = self.dense4(x)
        return x

