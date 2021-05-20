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

class DataGenerator(tf.keras.utils.Sequence):
        def __init__(self,
                     x, # nucleotide windows
                     y, # labels
                     p, # editing extents
                     batch_size,
                     data_augmentation=False,
                     occlusion=False,
                     label_smoothing=False,
                     target_occlusion=True,
                     shuffle=True):
                # number of examples for each class label

                # identify unedited and edited entries
                self.neg_idx = np.where(y == 0)[0]
                self.pos_idx = np.where(y == 1)[0]

                # outsample positives
                # if len(self.neg_idx) > len(self.pos_idx):
                #         folds = int(np.ceil(len(self.neg_idx)/len(self.pos_idx)))
                #         self.pos_idx = np.repeat(self.pos_idx, folds)
                # else:
                #         folds = int(np.ceil(len(self.pos_idx)/len(self.neg_idx)))
                #         self.neg_idx = np.repeat(self.neg_idx, folds)

                # number of examples for the minority class
                self.min_len = min([len(self.neg_idx), len(self.pos_idx)])

                # the batch_size is divided by the number of labels, as mini
                # batches are balanced
                self.batch_size = int(np.floor(batch_size / 2))

                self.occlusion = occlusion
                self.data_augmentation = data_augmentation
                self.label_smoothing = label_smoothing
                self.target_occlusion = target_occlusion
                self.shuffle = shuffle

                # shuffle data
                self._shuffle_data()

                # self.y = tf.one_hot(y, depth=2)
                self.data = tf.concat([x,
                                       tf.expand_dims(y, 1),
                                       tf.expand_dims(p, 1)], axis=1)

                # batch mask for occluding target position
                target_pos = 20
                win_len = 41
                self.target_mask = tf.one_hot(target_pos, depth=win_len)
                self.target_mask = tf.expand_dims(self.target_mask, 0)
                self.target_mask = tf.tile(self.target_mask, (batch_size, 1))
                self.target_mask = tf.cast(self.target_mask, tf.bool)
                self.target_mask = tf.logical_not(self.target_mask)

        def __len__(self):
                return int(np.floor(self.min_len / float(self.get_batch_size())))

        def get_batch_size(self):
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
                edited_X = tf.where(esites_mask, _NT2ID['C'], X)

                # replace sampled esites
                if self.data_augmentation:
                        # sample esites
                        random_mask = tf.random.uniform(X.shape)
                        cutoff = tf.random.uniform((X.shape[0], 1))
                        #sample_mask = tf.greater(random_mask, .5)
                        sample_mask = tf.greater(random_mask, cutoff)
                        sampled_esites = tf.math.logical_and(sample_mask, esites_mask)

                        edited_X = tf.where(sampled_esites, _NT2ID['T'], edited_X)

                return edited_X

        def occlude_random(self, X, cutoff=0.7):
                mask = tf.random.uniform(X.shape)

                return tf.where(mask > cutoff, X, -1)

        def occlude_block(self, X, maxlen=10):
                """
                Arg
                ---

                X: is a tensor whose shape is (n_wins, n_nucleotides)

                maxlen: integer indicating the maximum length of occluded regions
                """
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


        def _occlude(self, X):
                if self.occlusion:
                        if tf.random.uniform((1,))[0] > 0.5:
                                if tf.random.uniform((1,))[0] > 0.5:
                                        X = self.occlude_block(X)
                                else:
                                        X = self.occlude_random(X, cutoff=.5)

                # occlude target positions
                if tf.random.uniform((1,))[0] > 0.5:
                        X = tf.where(self.target_mask, X, _NT2ID['N'])

                return X

        def __getitem__(self, idx):
                from_rng = self.get_batch_size() * idx
                to_rng = self.get_batch_size() * (idx+1)
                idx = np.concatenate((self.neg_idx[from_rng:to_rng],
                                      self.pos_idx[from_rng:to_rng]))

                # retrieve windows
                slice = tf.nn.embedding_lookup(self.data, idx)
                X = slice[:, 0:41] # windows
                Y = slice[:,41:42] # labels
                P = slice[:,42:43] # editing extents

                # edit and occlude windows
                X = self._edit_wins(X)
                X_occ = self._occlude(X)

                # convert windows into one-hot representations
                X = tf.one_hot(tf.cast(X, tf.int32), depth=4)
                X_occ = tf.one_hot(tf.cast(X_occ, tf.int32), depth=4)

                return (X_occ, (X, Y, P))

        def _shuffle_data(self):
                """Shuffle windows associated to each label and calculate indexes for negative
                and positive examples."""
                self.neg_idx = tf.random.shuffle(self.neg_idx)
                self.pos_idx = tf.random.shuffle(self.pos_idx)

        def on_epoch_end(self):
                'Shuffle data after each epoch'
                if self.shuffle:
                        self._shuffle_data()

def prepare_dataset(infile,
                    augmentation=True,
                    label_smoothing=True,
                    occlude_target=True,
                    training_set_size=.8,
                    read_labels=True,
                    read_edexts=True,

                    batch_size=16):

        # x: wins  (num_wins, 41)
        # y: labels (num_wins, 1)
        # p: editing extents (num_wins, 1)
        x, y, p  = data_handler.read_windows(infile,
                                             read_labels=read_labels,
                                             read_edexts=read_edexts,
                                             occlude_target=False)

        # indices of negative and positive windows
        neg_idx = np.where(y == 0)[0]
        pos_idx = np.where(y == 1)[0]

        # # generate a train and valid sets
        # neg_train_idx, neg_valid_idx = data_handler.train_valid_split(neg_idx, percentage=training_set_size)
        # pos_train_idx, pos_valid_idx = data_handler.train_valid_split(pos_idx, percentage=training_set_size)

        # generate train/valid sets
        neg_train_idx = neg_idx.tolist()
        neg_valid_idx = np.random.choice(neg_idx, int(len(neg_idx) * (1-training_set_size))).tolist()
        pos_train_idx = pos_idx.tolist()
        pos_valid_idx = np.random.choice(pos_idx, int(len(pos_idx) * (1-training_set_size))).tolist()

        train_idx = neg_train_idx + pos_train_idx
        valid_idx = neg_valid_idx + pos_valid_idx

        x_train = x[train_idx,:].astype(np.float32)
        y_train = y[train_idx].astype(np.float32)
        p_train = p[train_idx].astype(np.float32)
        # x_train = x.astype(np.float32)
        # y_train = y.astype(np.float32)
        # p_train = p.astype(np.float32)

        x_valid = x[valid_idx,:].astype(np.float32)
        y_valid = y[valid_idx].astype(np.float32)
        p_valid = p[valid_idx].astype(np.float32)

        train_gen = DataGenerator(x_train,
                                  y_train,
                                  p_train,
                                  batch_size=batch_size,
                                  data_augmentation=augmentation,
                                  occlusion=augmentation,
                                  label_smoothing=label_smoothing,
                                  target_occlusion=occlude_target,
                                  shuffle=True)
        valid_gen = DataGenerator(x_valid,
                                  y_valid,
                                  p_valid,
                                  batch_size=batch_size,
                                  data_augmentation=False,
                                  occlusion=False,
                                  label_smoothing=False,
                                  target_occlusion=occlude_target,
                                  shuffle=True)

        return train_gen, valid_gen
