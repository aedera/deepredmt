import sys
import os

import tensorflow as tf
import tf_keras
import numpy as np

from . import data_handler as dh
from . import MODEL_FIN
from . import _NT2ID # nucleotide 2 index

def get_vector_representations(fin,
                               tf_model=None,
                               batch_size=512,
                               occlude_target=False):
    if tf_model is None:
        tf_model = MODEL_FIN

    # annotated editing sites as thymidines
    _NT2ID['E'] = _NT2ID['C']
    _NT2ID['e'] = _NT2ID['C']
    x = dh.read_windows(fin,
                        read_labels=False,
                        read_edexts=False,
                        occlude_target=occlude_target)[0]

    x = tf.one_hot(x, depth=4)
    model = tf.keras.models.load_model(tf_model, compile=False)

    # make a new model using only the encoder
    new_model = tf.keras.Model(inputs=model.get_layer('encoder').input,
                               outputs=model.get_layer('encoder').output)

    embeddings = new_model.predict(x, batch_size)

    return embeddings

def predict(fin, tf_model=None, batch_size=512, occlude_target=False):
    if tf_model is None:
        tf_model = MODEL_FIN

    # annotated editing sites as thymidines
    _NT2ID['E'] = _NT2ID['C']
    _NT2ID['e'] = _NT2ID['C']
    x = dh.read_windows(fin,
                        read_labels=False,
                        read_edexts=False,
                        occlude_target=occlude_target)[0]

    x = tf.one_hot(x, depth=4)
    model = tf.keras.models.load_model(tf_model, compile=False)
    x_rec, y_pred, _ = model.predict(x, batch_size=batch_size)

    # return label predictions
    return y_pred

def predict_from_fasta(fasin, tf_model=None, batch_size=512):
    if tf_model is None:
        tf_model = MODEL_FIN

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
    #model = keras.layers.TFSMLayer(tf_model, call_endpoint="serving_default")
    x_rec, y_pred, _ = model.predict(x, batch_size=batch_size, verbose=0)

    return raw_wins, y_pred
