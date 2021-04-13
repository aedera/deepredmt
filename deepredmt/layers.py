import tensorflow as tf
from tensorflow.keras import backend as K

class L2Normalization(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)
        self.axis = axis

    def get_config(self):
        config = super(L2Normalization, self).get_config()
        config.update({'axis': self.axis})
        return config

    def call(self, x):
        return tf.nn.l2_normalize(x, axis=self.axis)

class Conv1DTranspose(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=2,
                 output_padding=None,
                 padding='same', **kwargs):
        super(Conv1DTranspose, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.output_padding = output_padding
        self.padding = padding

    def build(self, input_shape):
        kinit = tf.keras.initializers.he_normal(seed=1234)

        self.conv = tf.keras.layers.Conv2DTranspose(
            filters=self.filters,
            kernel_size=(self.kernel_size, 1),
            strides=(self.strides, 1),
            output_padding=self.output_padding,
            kernel_initializer=kinit,
            bias_initializer='zeros',
            padding=self.padding)

    def get_config(self):
        config = super(Conv1DTranspose, self).get_config()

        config.update({'filters': self.filters})
        config.update({'kernel_size': self.kernel_size})
        config.update({'strides': self.strides})
        config.update({'output_padding': self.output_padding})
        config.update({'padding': self.padding})

        return config

    def call(self, x):
        x = tf.keras.layers.Lambda(lambda x: K.expand_dims(x, axis=2))(x)
        x = self.conv(x)
        x = tf.keras.layers.Lambda(lambda x: K.squeeze(x, axis=2))(x)

        return x
