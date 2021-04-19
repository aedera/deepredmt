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
from .models3 import Deepredmt, CAE
from . import data_handler

def pretrain(fin,
        num_hidden_units=5,
        batch_size=16,
        epochs=100,
        training_set_size=.8,
        save_model=False):

        x = data_handler.read_windows(
                fin,
                read_labels=False,
                read_edexts=False,
                occlude_target=False)[0]

        train_gen = utils.WinGenerator(x,
                                       batch_size=batch_size,
                                       data_augmentation=True,
                                       occlusion=True,
                                       shuffle=True)

        datetime_tag = datetime.datetime.now().strftime("%y%m%d-%H%M")
        log_dir = "./logs/"  'cae/' + datetime_tag
        callbacks = [
                tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                               histogram_freq=1),
                # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                #                                      factor=0.1,
                #                                      patience=3,
                #                                      verbose=1),
                # tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                #                                  patience=7,
                #                                  verbose=1)
        ]

        # breakpoint()
        # save the model with the best validation loss
        if save_model:
                model_file = "./models/" + 'cae/' + datetime_tag + ".tf"
                callbacks.append(
                        tf.keras.callbacks.ModelCheckpoint(
                                model_file,
                                save_best_only=True,
                                monitor='val_loss',
                                verbose=1)
                )

        win_shape = (41, 4)
        filters = [16, 32, 64, 128, 256, 512]
        model = CAE(filters, 5) #num_hidden_units)

        model.compile(tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()],
                      #run_eagerly=True,
        )

        #breakpoint()
        model.fit(train_gen,
                  epochs=50, #epochs,
                  callbacks=callbacks,
                  workers=16)

        model.save('autoencoder_v06.tf')
