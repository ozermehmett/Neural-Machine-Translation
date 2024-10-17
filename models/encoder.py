from tensorflow import keras
import tensorflow as tf

class Encoder(keras.models.Model):
    def __init__(self, vocab_size, num_hidden=256, num_embedding=256, batch_size=16):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        self.num_embedding = num_embedding
        self.embedding = keras.layers.Embedding(vocab_size, num_embedding)
        self.gru = keras.layers.GRU(num_hidden,
                                  return_sequences=True,
                                  recurrent_initializer='glorot_uniform',
                                  return_state=True)

    def call(self, x, hidden):
        embedded = self.embedding(x)
        rnn_out, hidden = self.gru(embedded, initial_state=hidden)
        return rnn_out, hidden

    def init_hidden(self):
        return tf.zeros(shape=(self.batch_size, self.num_hidden))
