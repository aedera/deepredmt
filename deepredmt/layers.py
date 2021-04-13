import math

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from src import utils

class L2Normalization(keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)
        self.axis = axis

    def get_config(self):
        config = super(L2Normalization, self).get_config()
        config.update({'axis': self.axis})
        return config

    def call(self, x):
        return tf.nn.l2_normalize(x, axis=self.axis)
