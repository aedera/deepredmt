import sys
import os

import sklearn.metrics

import tensorflow as tf
import numpy as np

from .predict import predict_from_fasta

def deepredmt():
    fasin = sys.argv[1]

    current_path = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(current_path, '../model')
    model_fin = os.path.join(model_dir, '210421-1205.tf')

    wins, preds = predict_from_fasta(fasin, model_fin)

    for i, k in enumerate(wins):
        print('{}\t{}\t{:.3f}'.format(k,
                                      ''.join(wins[k]),
                                      preds[i][0]))
