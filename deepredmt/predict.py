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

    # return label predictions
    return preds[1]

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
    preds = model.predict(x, batch_size=batch_size)

    return raw_wins, preds
