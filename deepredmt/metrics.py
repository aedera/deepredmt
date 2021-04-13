import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K

def sen_fn(y_true, y_pred):
        y_t = K.cast_to_floatx(K.argmax(y_true))
        y_p = K.cast_to_floatx(K.argmax(y_pred))
        return _sen_fn(y_t, y_p)

def _sen_fn(y_true, y_pred):
        agreements = K.sum(y_true * y_pred)
        sen = agreements / (K.sum(y_true) + K.epsilon())
        return sen

def spe_fn(y_true, y_pred):
        y_t = 1.0 - K.cast_to_floatx(K.argmax(y_true))
        y_p = 1.0 - K.cast_to_floatx(K.argmax(y_pred))
        return _sen_fn(y_t, y_p)

def pre_fn(y_true, y_pred):
        y_t = K.cast_to_floatx(K.argmax(y_true))
        y_p = K.cast_to_floatx(K.argmax(y_pred))
        return _pre_fn(y_t, y_p)

def _pre_fn(y_true, y_pred):
        # true positives
        true_pos = K.sum(y_true * y_pred)
        # false negatives
        compl_y_t = 1.0 - y_true
        false_neg = K.sum(compl_y_t * y_pred)
        return true_pos / (true_pos + false_neg)
