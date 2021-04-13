import tensorflow as tf

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
