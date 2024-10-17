from tensorflow import keras
import tensorflow as tf
from .attention import BahdanauAttention

class Decoder(keras.models.Model):
    def __init__(self, vocab_size, dec_dim=256, embedding_dim=256):
        super(Decoder, self).__init__()
        self.attn = BahdanauAttention(dec_dim)
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = keras.layers.GRU(dec_dim,
                                  recurrent_initializer='glorot_uniform',
                                  return_sequences=True,
                                  return_state=True)
        self.fc = keras.layers.Dense(vocab_size)

    def call(self, x, enc_hidden, enc_out):
        x = self.embedding(x)
        context, attn_weights = self.attn(enc_out, enc_hidden)
        x = tf.concat((tf.expand_dims(context, 1), x), -1)
        r_out, hidden = self.gru(x, initial_state=enc_hidden)
        out = tf.reshape(r_out, shape=(-1, r_out.shape[2]))
        return self.fc(out), hidden, attn_weights
