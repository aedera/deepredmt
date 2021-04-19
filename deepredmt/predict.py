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

    return preds

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
    y_pred = y_pred

    measures = []
    for t in np.arange(0., 1., .1):
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred >= t).ravel()
        re, pr, f1 = performance(y_true, y_pred >= t)
        measures.append([tn, fp, fn, tp, re, pr, f1])

    return measures
