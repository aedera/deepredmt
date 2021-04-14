import tensorflow as tf
import numpy as np
import sklearn.metrics

from . import data_handler
from . import _NT2ID # nucleotide 2 index

def predict(fin, tf_model, batch_size=512):
    # annotated editing sites as thymidines
    _NT2ID['E'] = _NT2ID['T']
    _NT2ID['e'] = _NT2ID['T']
    x = data_handler.read_windows(fin,
                                  read_labels=False,
                                  read_edexts=False)[0]
    x = tf.one_hot(x, depth=4)

    model = tf.keras.models.load_model(tf_model, compile='False')
    preds = model.predict(x, batch_size=batch_size)

    return preds[1]

def performance(y_true, y_pred):
    re = sklearn.metrics.recall_score(y_true, y_pred) #, average='micro')
    pr = sklearn.metrics.precision_score(y_true, y_pred) #, average='micro')
    f1 = sklearn.metrics.f1_score(y_true, y_pred) #, average='micro')

    return re, pr, f1

def pr_measures(fin, tf_model, batch_size=512):
    # annotated editing sites as thymidines
    _NT2ID['E'] = _NT2ID['T']
    _NT2ID['e'] = _NT2ID['T']
    x, y_true = data_handler.read_windows(fin,
                                          read_labels=True,
                                          read_edexts=False)
    x = tf.one_hot(x, depth=4)
    model = tf.keras.models.load_model(tf_model, compile='False')
    y_pred = model.predict(x, batch_size=batch_size)[1]
    y_pred = y_pred[:,0]

    measures = [performance(y_true > 0, y_pred >= t) for t in np.arange(0., 1., .1)]

    return measures
