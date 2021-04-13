import tensorflow as tf

HYPERPARAM = 0.0001

def classification_loss_fn(y_true,
                           y_pred,
                           ext_true,
                           label_smoothing=True):
        # label smoothing
        if label_smoothing:
                y_true = tf.math.multiply(ext_true, y_true)
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return tf.reduce_mean(loss)

def reconstruction_loss_fn(true, recon):
        recon_loss = tf.keras.losses.binary_crossentropy(true, recon)
        return tf.reduce_mean(recon_loss)

def regularization(params):
        return tf.add_n([ tf.nn.l2_loss(v) for v in params ]) * HYPERPARAM
