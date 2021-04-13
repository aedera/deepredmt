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

def Conv1DTranspose(input_tensor,
                    filters,
                    kernel_size,
                    strides=2,
                    output_padding=None,
                    padding='same'):
    x = tf.keras.layers.Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = tf.keras.layers.Conv2DTranspose(filters=filters,
                                        kernel_size=(kernel_size, 1),
                                        strides=(strides, 1),
                                        output_padding=output_padding,
                                        kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                                        bias_initializer='zeros',
                                        padding=padding)(x)
    x = tf.keras.layers.Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x
