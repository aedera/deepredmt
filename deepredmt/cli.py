import sys
import os

import sklearn.metrics

import tensorflow as tf
import numpy as np

from .predict import predict_from_fasta
from .predict import predict

from . import MODEL_FIN # path to trained model

def _predict_from_fasta(fasin):
    wins, preds = predict_from_fasta(fasin, MODEL_FIN)

    for i, k in enumerate(wins):
        lwin = wins[k][0:20]
        targ = wins[k][20]
        rwin = wins[k][21:]
        print('{}\t{}\t{}\t{}\t{:.3f}'.format(
            k,
            ''.join(lwin), targ, ''.join(rwin),
            preds[i][0]))

def deepredmt():
    if len(sys.argv) < 2:
        print('Fasta file not detected. Please indicate an input fasta file or a file containing 41-bp nucleotide windows.')
        return sys.exit(1)

    fin = sys.argv[1]

    # check if the input file is a fasta file or a file containing windows
    fline = open(fin).readline().rstrip()[0]
    if fline == '>':
        _predict_from_fasta(fin)
    elif fline != ' ':
        np.savetxt(sys.stdout.buffer, predict(fin, MODEL_FIN), fmt="%.2f")
    else:
        print('Input file is not recognizable')
