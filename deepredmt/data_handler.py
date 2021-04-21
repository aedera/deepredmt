# read homologs
import sys
import random
import numpy as np
import tensorflow as tf

from . import _NT2ID # nucleotide 2 index

def read_windows(infile,
                 read_labels=True,
                 read_edexts=True,
                 occlude_target=True):
    """read nucleotide windows along with their editing extents. Nucleotides are
    encoded as integers: 0:A 1:C 2:G 3:T 4:e 5:E
    """
    labels = []
    wins = [] # window sequences
    exts = []

    with open(infile) as f:
        for a in f:
            b = a.strip().split('\t')

            # nucleotide window
            win = list(b[0])
            # nucleotides to indexes
            win = [_NT2ID[n] for n in win]
            # occlude center position
            if occlude_target:
                win[len(win) // 2] = _NT2ID['N']
            wins.append(win)

            if read_labels:
                # label
                label = int(b[1])
                labels.append(label)

            if read_edexts:
                # editing extent
                ext = float(b[2])
                exts.append(ext)

        out = [np.asarray(wins)]
        if read_labels:
            out.append(np.asarray(labels))

        if read_edexts:
            out.append(np.asarray(exts))

        return out

def train_valid_split(wins, percentage=.8, seed=1234):
    idxs = list(range(len(wins)))
    rnd = random.Random()
    rnd.seed(seed)
    rnd.shuffle(idxs)
    thr = int(len(idxs) * percentage)
    train_idxs = idxs[0:thr]
    valid_idxs = idxs[thr:]

    return [wins[i] for i in train_idxs], [wins[i] for i in valid_idxs]


def read_fasta(fin):
    """Read a fasta file"""
    import re

    descline = re.compile('^>')

    seqs = {}
    with open(fin) as f:
        for a in f:
            b = a.strip()

            if descline.match(b):
                # with offset to remove the leading '>'
                seqname = b[1:]
                seqs[seqname] = ""
            else:
                seqs[seqname] += b

    return seqs


def extract_wins_from_fasta(fin):
    """Extract nucleotide windows from an input fasta file. Nucleotides are
encoded as integers.
    """
    win_len = 20
    seqs = read_fasta(fin)

    # wins is a dict
    # key: sequence name ! position, and window
    # value: nucleotide window where nucleotides are encoded by integers
    wins = {}
    for seqname, seq in seqs.items():
        lseq = list(seq)
        lseq = [nt.upper() for nt in lseq] # uppercase

        for i, t in enumerate(lseq):
            if t == 'C' or t == 'E': # cytidine or esite
                pos = i - win_len
                pos = 0 if pos < 0 else pos
                lwin = lseq[pos:i-1] # left
                rwin = lseq[i+1:i+1+win_len] # right

                lpad = ['N' for _ in range(win_len - len(lwin))]
                rpad = ['N' for _ in range(win_len - len(rwin))]

                win = lpad + lwin + [t] + rwin + rpad
                #win = leftpad + leftwin + [nt] + rightwin + rightpad

                key = '{}!{}'.format(seqname, i + 1)
                wins[key] = win

    return wins
