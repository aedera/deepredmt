import tensorflow as tf

from .layers import L2Normalization
from .layers import Conv1DTranspose
from .losses import reconstruction_loss_fn, classification_loss_fn


class Deepredmt(tf.keras.Model):
    def _build(input_shape, cae):
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs

        y = cae.encode(x)
        x_rec = cae.decode(y)

        # predict editing
        y = tf.keras.layers.Dense(
            1,
            kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
            bias_initializer=tf.keras.initializers.he_normal(seed=1234),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            bias_regularizer=tf.keras.regularizers.l2(1e-3))(y)
        y = tf.keras.layers.Activation('sigmoid', name='cla')(y)

        # final model
        return tf.keras.Model(inputs=inputs, outputs=[x_rec, y])

    def build(input_shape, num_hunits):
        filters = [16, 32, 64, 128, 256, 512]

        cae = CAE(filters, num_hunits)
        cae.build(input_shape)

        m = Deepredmt._build(input_shape, cae)
        model = Deepredmt(m.input, m.output, name='deepredmt')

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


class CAE(tf.keras.Model):
    """Convolutional AutoEncoder"""

    def __init__(self, filters, num_hunits, **kwargs):
        super(CAE, self).__init__(**kwargs)
        self.filters = filters
        self.num_hunits = num_hunits

    def build(self, input_shape):
        enco_layers = []
        deco_layers = []

        # window preprocessing
        for i in range(2):
            enco_layers.append(
                tf.keras.layers.Conv1D(
                    self.filters[0],
                    kernel_size=3,
                    strides=1,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                    bias_initializer=tf.keras.initializers.he_normal(seed=1234),
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                    bias_regularizer=tf.keras.regularizers.l2(1e-3),
                    padding='same')
            )
            enco_layers.append(
                tf.keras.layers.Activation('relu')
            )

        for f in self.filters[1:]:
            enco_layers.append(
                tf.keras.layers.Conv1D(
                    f,
                    kernel_size=3,
                    strides=2,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                    bias_initializer=tf.keras.initializers.he_normal(seed=1234),
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                    bias_regularizer=tf.keras.regularizers.l2(1e-3),
                    padding='same')
            )
            enco_layers.append(
                tf.keras.layers.Activation('relu')
            )

        enco_layers.append(
            tf.keras.layers.Flatten()
        )

        enco_layers.append(
            tf.keras.layers.Dense(
                self.num_hunits,
                name='embedder',
                use_bias=False,
                kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                bias_initializer=tf.keras.initializers.he_normal(seed=1234),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                bias_regularizer=tf.keras.regularizers.l2(1e-3))
        )

        deco_layers.append(
            tf.keras.layers.Dense(
                2*self.filters[-1],
                use_bias=False,
                kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                bias_initializer=tf.keras.initializers.he_normal(seed=1234),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                bias_regularizer=tf.keras.regularizers.l2(1e-3))
        )
        deco_layers.append(
            tf.keras.layers.Reshape((2, self.filters[-1]))
        )

        deco_layers.append(
            tf.keras.layers.Conv1DTranspose(
                self.filters[-2],
                kernel_size=2,
                strides=1,
                padding='valid',
                kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                bias_initializer=tf.keras.initializers.he_normal(seed=1234),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                bias_regularizer=tf.keras.regularizers.l2(1e-3),
                output_padding=None)
        )
        deco_layers.append(
            tf.keras.layers.Activation('relu')
        )

        deco_layers.append(
            tf.keras.layers.Conv1DTranspose(
                self.filters[-2],
                kernel_size=2,
                strides=2,
                padding='valid',
                kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                bias_initializer=tf.keras.initializers.he_normal(seed=1234),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                bias_regularizer=tf.keras.regularizers.l2(1e-3),
                output_padding=None)
        )
        deco_layers.append(
            tf.keras.layers.Activation('relu')
        )

        for i in range(3):
            deco_layers.append(
                tf.keras.layers.Conv1DTranspose(
                    self.filters[-i-4],
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                    bias_initializer=tf.keras.initializers.he_normal(seed=1234),
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                    bias_regularizer=tf.keras.regularizers.l2(1e-3),
                    output_padding=(0,))
            )
            deco_layers.append(
                tf.keras.layers.Activation('relu')
            )

        # reconstruction of sequences
        deco_layers.append(
            tf.keras.layers.Conv1D(
                4,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                bias_initializer=tf.keras.initializers.he_normal(seed=1234),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                bias_regularizer=tf.keras.regularizers.l2(1e-3))
        )
        deco_layers.append(
            tf.keras.layers.Activation('softmax', name='rec')
        )

        self.encoder = tf.keras.Sequential(
            [tf.keras.layers.InputLayer((41, 4))] + enco_layers,
            name='encoder'
        )

        self.decoder = tf.keras.Sequential(
            [tf.keras.layers.InputLayer((self.num_hunits))] + deco_layers,
            name='decoder'
        )

    def get_config(self):
        config = super(CAE, self).get_config()

        config.update({'filters': self.filters})
        config.update({'num_hunits': self.num_hunits})

        return config

    def call(self, x):
        z = self.encoder(x)
        y = self.decoder(z)

        return y

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
