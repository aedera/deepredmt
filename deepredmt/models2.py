import tensorflow as tf

from .layers import L2Normalization
from .layers import Conv1DTranspose
from .losses import reconstruction_loss_fn, classification_loss_fn


class CAE(tf.keras.Model):
    def _build(input_shape, num_hunits):
        filters = [16, 16, 32, 64, 128, 256, 512]

        #
        # Encoder
        #
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        # first convolutional layer
        x = tf.keras.layers.Conv1D(
            filters[0],
            3,
            strides=1,
            kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        # second convolutional layer
        x = tf.keras.layers.Conv1D(
            filters[1],
            3,
            strides=1,
            kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        # third convolutional layer
        x = tf.keras.layers.Conv1D(
            filters[2],
            3,
            strides=2,
            kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        # forth convolutional layer
        x = tf.keras.layers.Conv1D(
            filters[3],
            3,
            strides=2,
            kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        # fifth convolutional layer
        x = tf.keras.layers.Conv1D(
            filters[4],
            3,
            strides=2,
            kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            padding='valid')(x)
        x = tf.keras.layers.Activation('relu')(x)
        # sixth convolutional layer
        x = tf.keras.layers.Conv1D(
            filters[5],
            3,
            strides=1,
            kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            padding='valid')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv1D(
            filters[6],
            2,
            strides=1,
            kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            padding='valid')(x)
        x = tf.keras.layers.Activation('relu')(x)
        # flatten convolutional filters
        x = tf.keras.layers.Flatten()(x)
        y = tf.keras.layers.Dense(
            num_hunits,
            kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
            bias_initializer='zeros')(x)
        # l2-normalization
        #y = L2Normalization(axis=1)(y)
        #
        # decoder
        #
        x = tf.keras.layers.Dense(
            2*filters[6],
            kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
            bias_initializer='zeros')(y)
        x = tf.keras.layers.Reshape((2, filters[6]))(x)
        x = tf.keras.layers.Activation('relu')(x)
        # x = layers.ZeroPadding1D((0,1))(x)
        x = Conv1DTranspose(filters=filters[5],
                            kernel_size=2,
                            strides=1,
                            output_padding=None,
                            padding='valid')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = Conv1DTranspose(filters=filters[4],
                            kernel_size=3,
                            strides=1,
                            output_padding=(0, 0),
                            padding='valid')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = Conv1DTranspose(filters=filters[3],
                            kernel_size=3,
                            strides=2,
                            output_padding=(0, 0),
                            padding='valid')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = Conv1DTranspose(filters=filters[2],
                            kernel_size=3,
                            strides=2,
                            output_padding=(0, 0),
                            padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = Conv1DTranspose(filters=filters[1],
                            kernel_size=3,
                            strides=2,
                            output_padding=(0, 0),
                            padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = Conv1DTranspose(filters=filters[0],
                            kernel_size=3,
                            strides=1,
                            padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)

        # Last transposed convolutional layer used for reconstructing inputs
        x = Conv1DTranspose(filters=input_shape[1],
                            kernel_size=3,
                            strides=1,
                            padding='same')(x)
        x = tf.keras.layers.Activation('softmax', name='rec')(x)

        # softmax classifier
        z = tf.keras.layers.Dense(1,
                                  kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                                  bias_initializer='zeros')(y)
        z = tf.keras.layers.Activation('sigmoid', name='cla')(z)

        # final model
        return tf.keras.Model(inputs=inputs, outputs=[x, z])


    def build(input_shape, num_hunits, _label_smoothing=False):
        m = CAE._build(input_shape, num_hunits)
        model = CAE(m.input, m.output, name='deepredmt')

        optimizer = tf.keras.optimizers.Adam()

        model.compile(optimizer=optimizer,
                      loss=[
                          tf.keras.losses.CategoricalCrossentropy(),
                          tf.keras.losses.BinaryCrossentropy(),
                      ],
                      #run_eagerly=True,
        )

        return model

    @tf.function
    def train_step(self, batch):
        x, y  = batch

        with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(y, y_pred)

        # compute gradients
        variables = self.trainable_variables
        grad = tape.gradient(loss, variables)
        # update weights
        self.optimizer.apply_gradients(zip(grad, variables))
        self.compiled_metrics.update_state(y, y_pred, [])

        #         regularization = losses.regularization(model.trainable_variables)
        #         loss = recon_loss + class_loss + regularization
        # grads = tape.gradient(loss, model.trainable_variables)
        # # apply gradient to main model
        # opt.apply_gradients(zip(grads, model.trainable_variables))

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, batch):
        x, y = batch
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred, [])

        return {m.name: m.result() for m in self.metrics}
