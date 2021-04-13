import tensorflow as tf

from .layers import L2Normalization
from .layers import Conv1DTranspose

# CAE: Convolutional AutoEncoder
def CAE(input_shape, num_hunits, filters):
    #
    # Encoder
    #
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    # first convolutional layer
    x = tf.keras.layers.Conv1D(filters[0],
                               3,
                               strides=1,
                               kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                               bias_initializer='zeros',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                               padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    # second convolutional layer
    x = tf.keras.layers.Conv1D(filters[0],
                               3,
                               strides=1,
                               kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                               bias_initializer='zeros',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                               padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    # third convolutional layer
    x = tf.keras.layers.Conv1D(filters[1],
                               3,
                               strides=2,
                               kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                               bias_initializer='zeros',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                               padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    # forth convolutional layer
    x = tf.keras.layers.Conv1D(filters[2],
                               3,
                               strides=2,
                               kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                               bias_initializer='zeros',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                               padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    # fifth convolutional layer
    x = tf.keras.layers.Conv1D(filters[3],
                               3,
                               strides=2,
                               kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                               bias_initializer='zeros',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                               padding='valid')(x)
    x = tf.keras.layers.Activation('relu')(x)
    # sixth convolutional layer
    x = tf.keras.layers.Conv1D(filters[4],
                               3,
                               strides=1,
                               kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                               bias_initializer='zeros',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                               padding='valid')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv1D(filters[5],
                               2,
                               strides=1,
                               kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                               bias_initializer='zeros',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                               padding='valid')(x)
    x = tf.keras.layers.Activation('relu')(x)
    # flatten convolutional filters
    x = tf.keras.layers.Flatten()(x)
    y = tf.keras.layers.Dense(num_hunits,
                              kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                              bias_initializer='zeros')(x)
    # l2-normalization
    y = L2Normalization(axis=1)(y)
    #
    # decoder
    #
    x = tf.keras.layers.Dense(2*filters[5],
                              kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                              bias_initializer='zeros')(y)
    x = tf.keras.layers.Reshape((2, filters[5]))(x)
    x = tf.keras.layers.Activation('relu')(x)
    # x = layers.ZeroPadding1D((0,1))(x)
    x = Conv1DTranspose(x, filters=filters[4], kernel_size=2, strides=1, output_padding=None, padding='valid')
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv1DTranspose(x, filters=filters[3], kernel_size=3, strides=1, output_padding=(0, 0), padding='valid')
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv1DTranspose(x, filters=filters[2], kernel_size=3, strides=2, output_padding=(0, 0), padding='valid')
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv1DTranspose(x, filters=filters[1], kernel_size=3, strides=2, output_padding=(0, 0), padding='same')
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv1DTranspose(x, filters=filters[0], kernel_size=3, strides=2, output_padding=(0, 0), padding='same')
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv1DTranspose(x, filters=filters[0], kernel_size=3, strides=1, padding='same')
    x = tf.keras.layers.Activation('relu')(x)
    # Last transposed convolutional layer used for reconstructing inputs
    x = Conv1DTranspose(x, filters=input_shape[1], kernel_size=3, strides=1, padding='same')
    x = tf.keras.layers.Activation('sigmoid')(x)
    # softmax classifier
    z = tf.keras.layers.Dense(2,
                              kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                              bias_initializer='zeros')(y)
    z = tf.keras.layers.Activation('softmax')(z)

    # final model
    model = tf.keras.Model(inputs=inputs, outputs=[x, y, z])
    print(model.summary())
    return model
