import tensorflow as tf

class CAE(tf.keras.Model):
    """Convolutional AutoEncoder"""

    def __init__(self, filters, num_hunits, **kwargs):
        super(CAE, self).__init__(**kwargs)
        self.filters = filters
        self.num_hunits = num_hunits

    def build(self, input_shape):
        enco_layers = []
        deco_layers = []

        enco_layers.append(tf.keras.layers.BatchNormalization())

        # window preprocessing
        for i in range(3):
            enco_layers += [
                tf.keras.layers.Conv1D(
                    filters=self.filters[0],
                    kernel_size=3,
                    strides=1,
                    name='conv1d_0'+'_{}'.format(i),
                    kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                    bias_initializer=tf.keras.initializers.he_normal(seed=1234),
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                    bias_regularizer=tf.keras.regularizers.l2(1e-4),
                    padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
            ]

        enco_layers += [
            tf.keras.layers.Conv1D(
                self.filters[0],
                kernel_size=3,
                strides=1,
                name='conv1d_1',
                kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                bias_initializer=tf.keras.initializers.he_normal(seed=1234),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                bias_regularizer=tf.keras.regularizers.l2(1e-4),
                padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ]

        for i, f in enumerate(self.filters[1:]):
            name = "conv1d_{}".format(i+2)
            enco_layers += [
                tf.keras.layers.Conv1D(
                    f,
                    kernel_size=3,
                    strides=2,
                    name=name,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                    bias_initializer=tf.keras.initializers.he_normal(seed=1234),
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                    bias_regularizer=tf.keras.regularizers.l2(1e-4),
                    padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu')
            ]

        enco_layers.append(
            tf.keras.layers.Flatten()
        )

        enco_layers.append(
            tf.keras.layers.Dense(
                self.num_hunits,
                name='embedder',
                #use_bias=False,
                kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                bias_initializer=tf.keras.initializers.he_normal(seed=1234),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                bias_regularizer=tf.keras.regularizers.l2(1e-4))
        )

        deco_layers.append(
            tf.keras.layers.Dense(
                2*self.filters[-1],
                #use_bias=False,
                name='upsampler',
                kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                bias_initializer=tf.keras.initializers.he_normal(seed=1234),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                bias_regularizer=tf.keras.regularizers.l2(1e-4))
        )
        deco_layers.append(
            tf.keras.layers.Reshape((2, self.filters[-1]))
        )

        deco_layers.append(
            tf.keras.layers.Conv1DTranspose(
                self.filters[-2],
                kernel_size=2,
                strides=1,
                name='conv1d_transpose_0',
                padding='valid',
                kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                bias_initializer=tf.keras.initializers.he_normal(seed=1234),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                bias_regularizer=tf.keras.regularizers.l2(1e-4),
                output_padding=None)
        )
        deco_layers += [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ]

        deco_layers.append(
            tf.keras.layers.Conv1DTranspose(
                self.filters[-2],
                kernel_size=2,
                strides=2,
                padding='valid',
                name='conv1d_transpose_1',
                kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                bias_initializer=tf.keras.initializers.he_normal(seed=1234),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                bias_regularizer=tf.keras.regularizers.l2(1e-4),
                output_padding=None)
        )
        deco_layers += [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ]

        for i in range(3):
            deco_layers.append(
                tf.keras.layers.Conv1DTranspose(
                    self.filters[-i-4],
                    kernel_size=3,
                    strides=2,
                    name="conv1d_transpose_{}".format(i+2),
                    padding='same',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                    bias_initializer=tf.keras.initializers.he_normal(seed=1234),
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                    bias_regularizer=tf.keras.regularizers.l2(1e-4),
                    output_padding=(0,))
            )
            deco_layers += [
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu')
            ]

        # reconstruction of sequences
        deco_layers.append(
            tf.keras.layers.Conv1D(
                4,
                kernel_size=1,
                strides=1,
                padding='same',
                name='conv1d_transpose_5',
                kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
                bias_initializer=tf.keras.initializers.he_normal(seed=1234),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                bias_regularizer=tf.keras.regularizers.l1(1e-4))
        )
        deco_layers += [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('softmax', name='rec')
        ]

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

class Deepredmt(tf.keras.Model):
    def _build(input_shape, cae):
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs

        p = cae.encode(x)
        x_rec = cae.decode(p)

        y = tf.keras.layers.Dense(
            2,
            kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
            bias_initializer=tf.keras.initializers.he_normal(seed=1234),
            kernel_regularizer=tf.keras.regularizers.l2(1e-6),
            bias_regularizer=tf.keras.regularizers.l1(1e-6))(p)
        y = tf.keras.layers.Dense(
            1,
            kernel_initializer=tf.keras.initializers.he_normal(seed=1234),
            bias_initializer=tf.keras.initializers.he_normal(seed=1234),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            bias_regularizer=tf.keras.regularizers.l2(1e-4))(y)
        y = tf.keras.layers.Activation('sigmoid', name='cla')(y)

        # final model
        return tf.keras.Model(inputs=inputs, outputs=[x_rec, y, y])

    def build(input_shape, num_hunits):
        filters = [16, 32, 64, 128, 256, 512]

        cae = CAE(filters, num_hunits)
        cae.build(input_shape)

        m = Deepredmt._build(input_shape, cae)
        model = Deepredmt(m.input, m.output, name='deepredmt')

        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=optimizer,
                      loss = [
                          tf.keras.losses.CategoricalCrossentropy(),
                          tf.keras.losses.BinaryCrossentropy(),
                          tf.keras.losses.MeanAbsoluteError(),
                      ],
                      metrics = {
                          'cla': [
                              tf.keras.metrics.Precision(),
                              tf.keras.metrics.Recall(),
                              tf.keras.metrics.MeanSquaredError()
                          ],
                      },
                      loss_weights = [0.5, 1, 1],
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

        self.metrics[4].update_state(y[1], y_pred[1]) # precision
        self.metrics[5].update_state(y[1], y_pred[1]) # recall
        self.metrics[6].update_state(y[2], y_pred[2]) # MSE

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, batch):
        x, y = batch
        # y = y[1]
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred)

        self.compiled_metrics.update_state(y, y_pred, [])

        self.metrics[4].update_state(y[1], y_pred[1]) # precision
        self.metrics[5].update_state(y[1], y_pred[1]) # recall
        self.metrics[6].update_state(y[2], y_pred[2]) # MSE

        return {m.name: m.result() for m in self.metrics}
