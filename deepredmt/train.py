#!/usr/bin/env python3
import sys
import math
import datetime
#import tempfile

import tensorflow as tf
import numpy as np

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 42
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

from . import utils

def fit(fin,
        augmentation=True,
        label_smoothing=True,
        num_hidden_units=5,
        batch_size=16,
        epochs=100,
        training_set_size=.8,
        save_model=False):

        # prepare training and validation datasets
        train_gen, valid_gen = utils.prepare_dataset(
                fin,
                augmentation=augmentation,
                label_smoothing=label_smoothing,
                training_set_size=training_set_size,
                batch_size=batch_size)

        datetime_tag = datetime.datetime.now().strftime("%y%m%d-%H%M")
        log_dir = "./logs/"  'deepredmt/' + datetime_tag
        callbacks = [
                tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                               histogram_freq=1),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.1,
                                                     patience=3,
                                                     verbose=1),
                tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                 patience=7,
                                                 verbose=1)
        ]

        # save the model with the best validation loss
        if save_model:
                model_file = "./models/" + 'deepredmt/' + datetime_tag + ".tf"
                callbacks.append(
                        tf.keras.callbacks.ModelCheckpoint(
                                model_file,
                                save_best_only=True,
                                monitor='val_loss',
                                verbose=1)
                )
        from .models3 import CAE
        win_shape = (41, 4)
        model = CAE.build(win_shape, num_hidden_units)
        model.summary()

        model.fit(train_gen,
                  epochs=epochs,
                  validation_data=valid_gen,
                  callbacks=callbacks,
                  workers=16)
