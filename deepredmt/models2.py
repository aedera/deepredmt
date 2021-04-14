import tensorflow as tf

from .layers import L2Normalization
from .layers import Conv1DTranspose
from .losses import reconstruction_loss_fn, classification_loss_fn


class CAE(tf.keras.Model):
    def _build(input_shape, num_hunits):
        filters = [32, 64, 128, 256]

        inputs = tf.keras.Input(shape=input_shape)
        x = inputs

        # window preprocessing
        for i in range(4):
            x = tf.keras.layers.Conv1D(
                4**(i+2),
                kernel_size=3,
                dilation_rate=2**i,
                strides=1,
                kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                padding='same')(x)
            x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv1D(
            16,
            kernel_size=1,
            strides=1,
            kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)

        for f in filters:
            x = tf.keras.layers.Conv1D(
                f,
                kernel_size=3,
                strides=2,
                kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                padding='same')(x)
            x = tf.keras.layers.Activation('relu')(x)

        y = x # intermediate representation

        # Decoder
        x = tf.keras.layers.Conv1DTranspose(
            filters[-2],
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            output_padding=None)(y)
        x = tf.keras.layers.Activation('relu')(x)

        for f in reversed(filters[:-2]):
            x = tf.keras.layers.Conv1DTranspose(
                filters=f,
                kernel_size=2,
                strides=2,
                padding='same',
                kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                output_padding=(1))(x)
            x = tf.keras.layers.Activation('relu')(x)

        # reconstruction of sequences
        x = tf.keras.layers.Conv1DTranspose(
            4,
            kernel_size=2,
            strides=2,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            output_padding=(1))(x)
        x = tf.keras.layers.Activation('softmax', name='rec')(x)

        # predict editing
        y = tf.keras.layers.Flatten()(y)
        for i in range(7):
            y = tf.keras.layers.Dense(
                (3*256) // (2**i),
                kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                bias_initializer='zeros')(y)
            y = tf.keras.layers.BatchNormalization()(y)
            y = tf.keras.layers.Activation('relu')(y)
            y = tf.keras.layers.Dropout(.5)(y)

        # y = tf.keras.layers.Dense(
        #     num_hunits,
        #     kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
        #     bias_initializer='zeros',
        #     name='embedder')(y)
        #y = tf.keras.layers.Activation('relu')(y)
        y = tf.keras.layers.Dense(
            1,
            kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
            bias_initializer='zeros')(y)
        y = tf.keras.layers.Activation('sigmoid', name='cla')(y)

        # final model
        return tf.keras.Model(inputs=inputs, outputs=[x, y])

    def build(input_shape, num_hunits, _label_smoothing=False):
        m = CAE._build(input_shape, num_hunits)
        model = CAE(m.input, m.output, name='deepredmt')

        optimizer = tf.keras.optimizers.Adam()

        model.compile(optimizer=optimizer,
                      loss=[
                          tf.keras.losses.CategoricalCrossentropy(),
                          tf.keras.losses.BinaryCrossentropy(),
                      ],
                      metrics={
                          'cla': [tf.keras.metrics.Precision(),
                                  tf.keras.metrics.Recall()]
                      },
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
        self.compiled_metrics.update_state(y, y_pred, [])
        self.metrics[3].update_state(y[1], y_pred[1]) # precision
        self.metrics[4].update_state(y[1], y_pred[1]) # precision

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, batch):
        x, y = batch

        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred)

        self.compiled_metrics.update_state(y, y_pred, [])
        self.metrics[3].update_state(y[1], y_pred[1]) # precision
        self.metrics[4].update_state(y[1], y_pred[1]) # precision

        return {m.name: m.result() for m in self.metrics}
