import sys
import os

import sklearn.metrics

import tensorflow as tf
import numpy as np

from . import data_handler as dh
from . import _NT2ID # nucleotide 2 index

def predict(fin, tf_model, batch_size=512):
    # annotated editing sites as thymidines
    _NT2ID['E'] = _NT2ID['C']
    _NT2ID['e'] = _NT2ID['C']
    x = dh.read_windows(fin,
                        read_labels=False,
                        read_edexts=False)[0]
    x = tf.one_hot(x, depth=4)

    model = tf.keras.models.load_model(tf_model, compile='False')
    preds = model.predict(x, batch_size=batch_size)

    return preds

def predict_from_fasta(fasin, tf_model, batch_size=512):
    # annotated editing sites as thymidines
    raw_wins = dh.extract_wins_from_fasta(fasin)

    # encode nucleotide as integer
    nt2id = _NT2ID.copy()
    nt2id['E'] = nt2id['C'] # replace esites for cytidines

    wins = {}
    for k, w in raw_wins.items():
        wins[k] = []

        for n in w:
            if n in nt2id:
                encoding = nt2id[n]
            else:
                # encode unknown nucleotides as 'N'
                encoding = nt2id['N']
            wins[k].append(encoding)

    # encode windows as one-hot vectors
    x = np.asarray(list(wins.values()))
    x = tf.one_hot(x, depth=4)

    model = tf.keras.models.load_model(tf_model, compile='False')
    preds = model.predict(x, batch_size=batch_size)[1]

    return raw_wins, preds

def performance(y_true, y_pred):
    re = sklearn.metrics.recall_score(y_true, y_pred)
    pr = sklearn.metrics.precision_score(y_true, y_pred)
    f1 = sklearn.metrics.f1_score(y_true, y_pred)

    return re, pr, f1

def pr_curve(fin, tf_model, batch_size=512):
    # annotated editing sites as thymidines
    _NT2ID['E'] = _NT2ID['C']
    _NT2ID['e'] = _NT2ID['C']
    x, y_true = dh.read_windows(fin,
                                read_labels=True,
                                read_edexts=False)
    x = tf.one_hot(x, depth=4)
    model = tf.keras.models.load_model(tf_model, compile='False')
    y_pred = model.predict(x, batch_size=batch_size)[1]
    y_pred = y_pred

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_pred)
    auc = sklearn.metrics.auc(recall, precision)

    for t in np.arange(0., 1.01, .01):
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred >= t).ravel()
        re, pr, f1 = performance(y_true, y_pred >= t)

        print('{:.2f}\t{:d}\t{:d}\t{:d}\t{:d}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.2f}'.format(
            t,
            tn, fp, fn, tp,
            re, pr, f1, auc))
