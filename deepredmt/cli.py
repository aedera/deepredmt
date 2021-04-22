import sys
import os

import sklearn.metrics

import tensorflow as tf
import numpy as np

from .predict import predict_from_fasta

def deepredmt():
    if len(sys.argv) < 2:
        print('Fasta file not detected. Please indicate a fasta file path.')
        return sys.exit(1)

    fasin = sys.argv[1]

    current_path = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(current_path, './model')
    model_fin = os.path.join(model_dir, '210421-1205.tf')

    wins, preds = predict_from_fasta(fasin, model_fin)

    for i, k in enumerate(wins):
        lwin = wins[k][0:20]
        targ = wins[k][20]
        rwin = wins[k][21:]
        print('{}\t{}\t{}\t{}\t{:.3f}'.format(
            k,
            ''.join(lwin), targ, ''.join(rwin),
            preds[i][0]))
