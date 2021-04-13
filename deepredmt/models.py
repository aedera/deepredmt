import math

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers

from src.layers import L2Normalization

from tensorflow.keras.layers import Conv2DTranspose

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, output_padding=None, padding='same'):
    x = layers.Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters,
                        kernel_size=(kernel_size, 1),
                        strides=(strides, 1),
                        output_padding=output_padding,
                        kernel_initializer=keras.initializers.he_normal(seed=1234),
                        bias_initializer='zeros',
                       padding=padding)(x)
    x = layers.Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

# CAE: Convolutional AutoEncoder
def CAE(input_shape, num_hunits, filters):
    #
    # Encoder
    #
    inputs = Input(shape=input_shape)
    x = inputs
    # first convolutional layer
    x = keras.layers.Conv1D(filters[0],
                            3,
                            strides=1,
                            kernel_initializer=keras.initializers.he_normal(seed=1234),
                            bias_initializer='zeros',
                            kernel_regularizer=keras.regularizers.l2(1e-4),
                            padding='same')(x)
    x = Activation('relu')(x)
    # second convolutional layer
    x = keras.layers.Conv1D(filters[0],
                            3,
                            strides=1,
                            kernel_initializer=keras.initializers.he_normal(seed=1234),
                            bias_initializer='zeros',
                            kernel_regularizer=keras.regularizers.l2(1e-4),
                            padding='same')(x)
    x = Activation('relu')(x)
    # third convolutional layer
    x = keras.layers.Conv1D(filters[1],
                            3,
                            strides=2,
                            kernel_initializer=keras.initializers.he_normal(seed=1234),
                            bias_initializer='zeros',
                            kernel_regularizer=keras.regularizers.l2(1e-4),
                            padding='same')(x)
    x = Activation('relu')(x)
    # forth convolutional layer
    x = keras.layers.Conv1D(filters[2],
                            3,
                            strides=2,
                            kernel_initializer=keras.initializers.he_normal(seed=1234),
                            bias_initializer='zeros',
                            kernel_regularizer=keras.regularizers.l2(1e-4),
                            padding='same')(x)
    x = Activation('relu')(x)
    # fifth convolutional layer
    x = keras.layers.Conv1D(filters[3],
                            3,
                            strides=2,
                            kernel_initializer=keras.initializers.he_normal(seed=1234),
                            bias_initializer='zeros',
                            kernel_regularizer=keras.regularizers.l2(1e-4),
                            padding='valid')(x)
    x = Activation('relu')(x)
    # sixth convolutional layer
    x = keras.layers.Conv1D(filters[4],
                            3,
                            strides=1,
                            kernel_initializer=keras.initializers.he_normal(seed=1234),
                            bias_initializer='zeros',
                            kernel_regularizer=keras.regularizers.l2(1e-4),
                            padding='valid')(x)
    x = Activation('relu')(x)
    x = keras.layers.Conv1D(filters[5],
                            2,
                            strides=1,
                            kernel_initializer=keras.initializers.he_normal(seed=1234),
                            bias_initializer='zeros',
                            kernel_regularizer=keras.regularizers.l2(1e-4),
                            padding='valid')(x)
    x = Activation('relu')(x)
    # flatten convolutional filters
    x = Flatten()(x)
    y = Dense(num_hunits,
              kernel_initializer=keras.initializers.he_normal(seed=1234),
              bias_initializer='zeros')(x)
    # l2-normalization
    y = L2Normalization(axis=1)(y)
    #
    # decoder
    #
    x = Dense(2*filters[5],
              kernel_initializer=keras.initializers.he_normal(seed=1234),
              bias_initializer='zeros')(y)
    x = layers.Reshape((2, filters[5]))(x)
    x = Activation('relu')(x)
    # x = layers.ZeroPadding1D((0,1))(x)
    x = Conv1DTranspose(x, filters=filters[4], kernel_size=2, strides=1, output_padding=None, padding='valid')
    x = Activation('relu')(x)
    x = Conv1DTranspose(x, filters=filters[3], kernel_size=3, strides=1, output_padding=(0, 0), padding='valid')
    x = Activation('relu')(x)
    x = Conv1DTranspose(x, filters=filters[2], kernel_size=3, strides=2, output_padding=(0, 0), padding='valid')
    x = Activation('relu')(x)
    x = Conv1DTranspose(x, filters=filters[1], kernel_size=3, strides=2, output_padding=(0, 0), padding='same')
    x = Activation('relu')(x)
    x = Conv1DTranspose(x, filters=filters[0], kernel_size=3, strides=2, output_padding=(0, 0), padding='same')
    x = Activation('relu')(x)
    x = Conv1DTranspose(x, filters=filters[0], kernel_size=3, strides=1, padding='same')
    x = Activation('relu')(x)
    # Last transposed convolutional layer used for reconstructing inputs
    x = Conv1DTranspose(x, filters=input_shape[1], kernel_size=3, strides=1, padding='same')
    x = Activation('sigmoid')(x)
    # softmax classifier
    z = Dense(2,
              kernel_initializer=keras.initializers.he_normal(seed=1234),
              bias_initializer='zeros')(y)
    z = Activation('softmax')(z)
    # final model
    model = Model(inputs=inputs, outputs=[x, y, z])
    print(model.summary())
    return model
