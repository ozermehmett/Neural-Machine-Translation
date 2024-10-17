from tensorflow import keras
import tensorflow as tf

class BahdanauAttention(keras.models.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = keras.layers.Dense(units)
        self.W2 = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)

    def call(self, encoder_out, hidden):
        hidden = tf.expand_dims(hidden, axis=1)
        score = self.V(tf.nn.tanh(self.W1(encoder_out) + self.W2(hidden)))
        attn_weights = tf.nn.softmax(score, axis=1)
        context = attn_weights * encoder_out
        context = tf.reduce_sum(context, axis=1)
        return context, attn_weights
