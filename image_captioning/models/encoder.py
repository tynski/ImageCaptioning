import tensorflow as tf


class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

    # Input shape: N - D tensor with shape: (batch_size, ..., input_dim).The
    # most common situation would be a 2D input with shape(batch_size, input_dim).
