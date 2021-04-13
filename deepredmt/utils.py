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

#from . import _NT2ID
from . import data_handler

class DataGenerator():
        def __init__(self,
                     data,
                     batch_size,
                     data_augmentation=False,
                     occlusion=False,
                     shuffle=True):
                self.data = data
                # N, A, C, G, T. Minus one because states e and E are not considered
                self.num_states = len(_NT2ID) - 2
                # number of examples for each class label
                self.label_counter = np.unique(self.data[:,0], return_counts=True)[1]
                self.num_labels = len(self.label_counter)
                # number of examples for the minority class
                self.min_len = min(self.label_counter)
                # the batch_size is divided by the number of labels, as mini
                # batches are balanced
                self.batch_size = int(np.floor(batch_size / self.num_labels))
                self.occlusion = occlusion
                self.data_augmentation = data_augmentation
                self.shuffle = shuffle
                # shuffle data
                self._shuffle_data()

        def __len__(self):
                return int(np.floor(self.min_len / float(self.get_batch_size())))

        def get_batch_size(self):
                # batch augmentation is applied when occlusion is True, so
                # batch_size is halved
                #return int(self.batch_size / 2) if self.occlusion else self.batch_size
                return self.batch_size

        def generate_random_mask(self, num_wins, seq_length, start_pos, end_pos, min_length, max_length):
                occlusion_lengths = tf.random.uniform((num_wins,),  minval=min_length, maxval=max_length, dtype=tf.int32)
                occlusion_lengths = tf.concat([ [occlusion_lengths - i] for i in range(max_length) ], 0)
                occlusion_lengths = tf.transpose(occlusion_lengths)
                occlusion_lengths = tf.maximum(occlusion_lengths, 0)
                #
                positions = tf.random.uniform((num_wins,), minval=start_pos, maxval=end_pos, dtype=tf.int32)
                positions = tf.transpose(tf.broadcast_to(positions, (max_length, num_wins)))
                occlusion_regions = tf.reduce_sum(tf.one_hot(occlusion_lengths + positions, seq_length), 1)
                occlusion_regions = tf.cast(tf.greater(occlusion_regions,0), tf.float32)
                return occlusion_regions

        def generate_sided_mask(self, side, batch_sz, min_length, max_length):
                num_wins = batch_sz[0]
                seq_length = batch_sz[1]
                target_pos = int(np.floor(seq_length / 2))
                if side < 0:
                        mask = self.generate_random_mask(num_wins,
                                                         seq_length,
                                                         0,
                                                         target_pos,
                                                         min_length,
                                                         max_length)
                else:
                        mask = self.generate_random_mask(num_wins,
                                                         seq_length,
                                                         target_pos,
                                                         seq_length,
                                                         min_length,
                                                         max_length)
                return mask

        def _occlude_batch(self, batch, min_length=1, max_length=5):
                side = random.sample([-1, 1], 1)[0]
                mask = self.generate_sided_mask(side,
                                                batch.shape,
                                                min_length,
                                                max_length)
                # second round of occlusions
                mask_2 = self.generate_sided_mask(side * -1,
                                                  batch.shape,
                                                  min_length,
                                                  max_length)
                indicators = tf.cast(tf.random.uniform((batch.shape[0],), minval=0, maxval=2, dtype=tf.int64), tf.float32)
                indicators = tf.expand_dims(indicators, 1)
                mask_2 = tf.multiply(indicators, mask_2)
                # combine both masks
                mask = tf.math.add(mask, mask_2)
                #
                mask = tf.cast(tf.logical_not(tf.cast(mask, tf.bool)), tf.float32)
                occluded_batch = tf.multiply(mask, batch)
                return occluded_batch

         # windows in the batch are randomly edited
        def _edit_wins(self, X):
                """Edit window if data_augmentation is true by converting at random esites in
                cytidines or thymidines. Otherwise, convert esites
                into cytidines.
                """
                # mask target position
                target_pos = int(X.shape[1] / 2)
                X[:, target_pos] = _NT2ID['C']
                #
                # convert esites to cytidines if data_augmentation is False
                #
                if not self.data_augmentation:
                        for nt in (['E', -3.], ['e', -4.]):
                                mask = tf.equal(X, _NT2ID[nt[0]])
                                nt2Cmask = nt[1] * tf.cast(mask, tf.float32)
                                X = tf.math.add(X, nt2Cmask)
                        return X
                #
                # Random editing if data_augmentation is True
                #
                # get mask with esites
                Emask = tf.equal(X, _NT2ID['E'])
                # randomly convert strong esites into either thymidines or cytidines
                EtoTmask = tf.greater(tf.multiply(tf.cast(Emask, tf.float32), tf.random.uniform(Emask.shape)), .5)
                EtoCmask = tf.logical_and(Emask, tf.logical_not(EtoTmask))
                # -1 to move E (5) to T (4)
                EtoTmask = -1. * tf.cast(EtoTmask, tf.float32)
                # -3 to move E (5) to C (2)
                EtoCmask = -3. * tf.cast(EtoCmask, tf.float32)
                X = tf.math.add(X, EtoTmask)
                X = tf.math.add(X, EtoCmask)

                return X

        def __getitem__(self, idx):
                from_rng = self.get_batch_size() * idx; to_rng = self.get_batch_size() * (idx+1)
                idx = np.concatenate((self.neg_idx[from_rng:to_rng],
                                      self.pos_idx[from_rng:to_rng]))
                # retrieve labels
                Y = self.data[idx, 0]
                X = np.array(list(map(list, self.data[idx, 1])))
                # retrieve windows and make data augmentation
                X = self._edit_wins(X.copy())
                # batch augmentation by occluding windows
                if self.occlusion:
                        occluded_X = self._occlude_batch(X,
                                                         min_length=1,
                                                         max_length=5)
                else:
                        occluded_X = []

                X = tf.keras.utils.to_categorical(X, num_classes=self.num_states)
                # remove N nucleotides which is the state 0 (i.e. the first
                # element in the one-hot vector). Note that occluded regions
                # are also marked by Ns
                X = tf.slice(X, [0, 0, 1], [X.shape[0], X.shape[1], X.shape[2] - 1])
                Y = tf.convert_to_tensor(keras.utils.to_categorical(Y, num_classes=self.num_labels))
                Y = tf.cast(Y, tf.float32)
                # retrieve editing extents (used for label smoothing)
                Z = np.array(list(map(list, self.data[idx, 2])))
                Z = tf.convert_to_tensor(Z)
                Z = tf.cast(Z, tf.float32)
                # minibatch augmentation
                if self.occlusion:
                        occluded_X = tf.keras.utils.to_categorical(occluded_X, num_classes=self.num_states)
                        occluded_X = tf.slice(occluded_X, [0, 0, 1], [occluded_X.shape[0], occluded_X.shape[1], occluded_X.shape[2] - 1])
                        # denoising autoencoder (no occlusion)
                        W = X
                        X = occluded_X
                        # X = tf.concat([X, occluded_X], 0)
                        # Y = tf.cast(tf.tile(Y, [2,1]), tf.float32)
                        # Z = tf.cast(tf.tile(Z, [2,1]), tf.float32)
                else:
                        W = X
                return X, Y, Z, W

        def _shuffle_data(self):
                """Shuffle windows associated to each label and calculate indexes for negative
                and positive examples."""
                self.data = np.random.permutation(self.data)
                self.neg_idx = np.where(self.data[:,0] == 0)[0]
                self.pos_idx = np.where(self.data[:,0] == 1)[0]

        def on_epoch_end(self):
                'Shuffle data after each epoch'
                if self.shuffle:
                        self._shuffle_data()

def prepare_dataset(infile, augmentation=True):
        data  = data_handler.read_windows(infin)

        # generate a train and valid set for each label
        neg_train_idx, neg_valid_idx = data_handler.train_valid_split(np.where(data[:,0] == 0)[0])
        pos_train_idx, pos_valid_idx = data_handler.train_valid_split(np.where(data[:,0] == 1)[0])

        train_idx = neg_train_idx + pos_train_idx
        valid_idx = neg_valid_idx + pos_valid_idx

        train_set = np.asarray([data[i] for i in train_idx])
        valid_set = np.asarray([data[i] for i in valid_idx])

        train_gen = utils.DataGenerator(train_set,
                                        batch_size=16,
                                        data_augmentation=augmentation,
                                        occlusion=augmentation,
                                        shuffle=True)
        valid_gen = utils.DataGenerator(valid_set,
                                        batch_size=16,
                                        data_augmentation=False,
                                        occlusion=False,
                                        shuffle=True)

        return train_gen, valid_gen
