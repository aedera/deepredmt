import math
import copy

import tensorflow as tf

seed_value= 42
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

from . import _NT2ID # nucleotide 2 index
from . import data_handler

class DataGenerator():
        def __init__(self,
                     x, # nucleotide windows
                     y, # labels
                     p, # editing extents
                     batch_size,
                     data_augmentation=False,
                     occlusion=False,
                     shuffle=True):
                # N, A, C, G, T. Minus one because states e and E are not considered
                self.num_states = len(_NT2ID) - 2
                # number of examples for each class label
                self.label_counter = np.unique(y, return_counts=True)[1]
                self.num_labels = len(self.label_counter)
                # number of examples for the minority class
                self.min_len = min(self.label_counter)
                # the batch_size is divided by the number of labels, as mini
                # batches are balanced
                self.batch_size = int(np.floor(batch_size / self.num_labels))
                self.occlusion = occlusion
                self.data_augmentation = data_augmentation
                self.shuffle = shuffle

                # identify unedited and edited entries
                self.neg_idx = np.where(y == 0)[0]
                self.pos_idx = np.where(y == 1)[0]

                # shuffle data
                self._shuffle_data()

                # self.y = tf.one_hot(y, depth=2)
                # breakpoint()

                self.data = tf.concat([x,
                                       tf.one_hot(y, depth=2),
                                       p], axis=1)
        def __len__(self):
                return int(np.floor(self.min_len / float(self.get_batch_size())))

        def get_batch_size(self):
                # batch augmentation is applied when occlusion is True, so
                # batch_size is halved
                #return int(self.batch_size / 2) if self.occlusion else self.batch_size
                return self.batch_size

        def _edit_wins(self, X):
                """If data_augmentation is true, this method randomly edits esites in each
                window by transforming cytidines into thymidines. Otherwise,
                all esites are transformed into cytidines.
                """

                # mask indicating where there are esites
                esites_mask = tf.math.logical_or(tf.equal(X, _NT2ID['E']),
                                                 tf.equal(X, _NT2ID['e']))

                # replace all esites by cytidines
                new_X = tf.where(esites_mask, _NT2ID['C'], X)

                # replace sampled esites
                if self.data_augmentation:
                        # sample esites
                        random_mask = tf.random.uniform(X.shape)
                        sample_mask = tf.greater(random_mask, .5)
                        sampled_esites = tf.math.logical_and(sample_mask, esites_mask)

                        new_X = tf.where(sampled_esites, _NT2ID['T'], new_X)

                return new_X

        def occlude(self, X, maxlen=5):
                """
                Arg
                ---

                X: is a tensor whose shape is (n_wins, n_nucleotides)

                maxlen: integer indicating the maximum length of occluded regions
                """
                maxlen = 5

                # length of the occluded regions
                occlen = tf.random.uniform((1,),
                                           minval=0,
                                           maxval=maxlen,
                                           dtype=tf.int32).numpy()[0]
                if occlen == 0:
                        return X

                # sample a seed position in which windows to be occluded
                #
                # [[1],
                #  [7],
                #  [3]]
                seeds = tf.random.uniform((X.shape[0], 1),
                                         minval=0,
                                         maxval=X.shape[1] - occlen,
                                         dtype=tf.int32)
                # repeat the seeds to match the length of occlusions
                #
                # [[1, 1, 1],
                #  [7, 7, 7],
                #  [3, 3 ,3]]
                mat_seeds = tf.repeat(seeds, occlen, axis=1)

                # generate offset
                # [0, 1, 2]
                offsets = tf.constant(range(occlen))
                # repeat offset to match seeds
                #
                # [[0, 1, 2],
                #  [0, 1, 2],
                #  [0, 1, 2]]
                mat_offsets = tf.repeat([offsets], X.shape[0], axis=0)

                occ_mask = tf.one_hot(mat_seeds + mat_offsets, depth=X.shape[1])
                occ_mask = tf.reduce_sum(occ_mask, 1)
                occ_mask = tf.cast(occ_mask, tf.bool)

                return tf.where(occ_mask, -1, X)

        def __getitem__(self, idx):
                from_rng = self.get_batch_size() * idx
                to_rng = self.get_batch_size() * (idx+1)
                idx = np.concatenate((self.neg_idx[from_rng:to_rng],
                                      self.pos_idx[from_rng:to_rng]))

                # retrieve windows
                slice = tf.nn.embedding_lookup(self.data, idx)
                X, Y, Z = slice[:, 0:41], slice[:,41:43], slice[:,43:45]

                # edit and occlude windows
                X = self._edit_wins(X)
                X_occ = self.occlude(X) if self.occlusion else X

                # convert windows into one-hot representations
                X = tf.one_hot(tf.cast(X, tf.int32), depth=4)
                X_occ = tf.one_hot(tf.cast(X_occ, tf.int32), depth=4)

                return X, Y, Z, X_occ

        def _shuffle_data(self):
                """Shuffle windows associated to each label and calculate indexes for negative
                and positive examples."""
                self.neg_idx = tf.random.shuffle(self.neg_idx)
                self.pos_idx = tf.random.shuffle(self.pos_idx)

        def on_epoch_end(self):
                'Shuffle data after each epoch'
                if self.shuffle:
                        self._shuffle_data()

def prepare_dataset(infile, augmentation=True, batch_size=16):
        # x: wins
        # y: labels
        # p: editing extents
        x, y, p  = data_handler.read_windows(infile)

        # generate a train and valid set for each label
        neg_train_idx, neg_valid_idx = data_handler.train_valid_split(np.where(y == 0)[0])
        pos_train_idx, pos_valid_idx = data_handler.train_valid_split(np.where(y == 1)[0])

        train_idx = neg_train_idx + pos_train_idx
        valid_idx = neg_valid_idx + pos_valid_idx

        x_train = x[train_idx,:].astype(np.float32)
        y_train = y[train_idx].astype(np.float32)
        p_train = p[train_idx].astype(np.float32)

        x_valid = x[valid_idx,:].astype(np.float32)
        y_valid = y[valid_idx].astype(np.float32)
        p_valid = p[valid_idx].astype(np.float32)

        train_gen = DataGenerator(x_train,
                                  y_train,
                                  p_train,
                                  batch_size=batch_size,
                                  data_augmentation=augmentation,
                                  occlusion=augmentation,
                                  shuffle=True)
        valid_gen = DataGenerator(x_valid,
                                  y_valid,
                                  p_valid,
                                  batch_size=batch_size,
                                  data_augmentation=False,
                                  occlusion=False,
                                  shuffle=True)

        return train_gen, valid_gen
