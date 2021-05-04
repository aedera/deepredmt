#!/usr/bin/env python3
seed_value= 42

import os
import sys
import math
import datetime
import random

import tensorflow as tf
import numpy as np

os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

from . import utils
from .models import Deepredmt

class WeightCheckpoint(tf.keras.callbacks.Callback):
        def __init__(self):
                super(WeightCheckpoint, self).__init__()
                self.best = np.Inf
                self.best_weights = None

        def on_epoch_end(self, epoch, logs=None):
                current = logs.get('val_loss')

                if current < self.best:
                        self.best = current
                        self.weights = self.model.get_weights()
                        print('\nCheckpointing weights val_loss= {0:6.4f}'.format(self.best))
                else:
                        print('\nBest val_loss {0:6.4f}'.format(self.best))

class MyEarlyStopping(tf.keras.callbacks.Callback):
        def __init__(self, cutoff):
                super(MyEarlyStopping, self).__init__()
                self.cutoff = cutoff

        def on_epoch_end(self, epoch, logs=None):
                if self.model.optimizer.learning_rate < self.cutoff:
                        self.model.stop_training = True
                        print('\nStop learning as learning rate is below the threshold\n')

def scheduler(epoch, lr):
        if epoch < 10:
                return (lr * (epoch+1))/10
        else:
                return lr

def get_callbacks(datetime_tag):
        log_dir = "./logs/"  'deepredmt/' + datetime_tag
        callbacks = [
                tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                               histogram_freq=1),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.1,
                                                     patience=3,
                                                     verbose=1),
                WeightCheckpoint(),
                MyEarlyStopping(1e-15),
                # tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                #                                  patience=4,
                #                                  verbose=1),
                # tf.keras.callbacks.LearningRateScheduler(stop_learning_based_on_lr,
                #                                          verbose=0)
        ]

        return callbacks


def fit(fin,
        augmentation=True,
        num_hidden_units=5,
        batch_size=16,
        epochs=100,
        training_set_size=.8):
        # prepare training and validation datasets
        train_gen, valid_gen = utils.prepare_dataset(
                fin,
                augmentation=augmentation,
                label_smoothing=True,
                training_set_size=training_set_size,
                occlude_target=False,
                batch_size=batch_size
        )
        #breakpoint()
        win_shape = (41, 4)
        model = Deepredmt.build(win_shape, num_hidden_units)
        model.get_layer('encoder').summary()
        model.summary()

        datetime_tag = datetime.datetime.now().strftime("%y%m%d-%H%M")
        callbacks = get_callbacks(datetime_tag)
        model.fit(train_gen,
                  epochs=epochs,
                  validation_data=valid_gen,
                  callbacks=callbacks,
                  workers=16)

        # save best model
        best_weights = [c for c in callbacks if type(c).__name__ == "WeightCheckpoint"][0].weights
        model.set_weights(best_weights)

        model_file = "./models/" + 'deepredmt/' + datetime_tag + ".tf"
        model.save(model_file)
        print('Best model saved.')

        return True
