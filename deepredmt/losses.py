import tensorflow as tf

HYPERPARAM = 0.0001

def classification_loss_fn(class_true,
                           class_pred,
                           ext_true,
                           label_smoothing=True):
        # label smoothing
        if label_smoothing:
                class_true = ext_true * class_true
        class_loss = tf.keras.losses.categorical_crossentropy(class_true, class_pred)
        return tf.reduce_mean(class_loss)

def reconstruction_loss_fn(true, recon):
        recon_loss = tf.keras.losses.binary_crossentropy(true, recon)
        return tf.reduce_mean(recon_loss)

def regularization(params):
        return tf.add_n([ tf.nn.l2_loss(v) for v in params ]) * HYPERPARAM
