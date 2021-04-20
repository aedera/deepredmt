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
from .models3 import CAE, Deepredmt

# class CustomCallback(tf.keras.callbacks.Callback):
#         def on_epoch_end(self, epoch, logs=None):
#                 # keys = list(logs.keys())
#                 # print("End epoch {} of training; got log keys: {}".format(epoch, keys))
#                 if epoch + 1 >= 3 and not self.model.get_layer('encoder').trainable:
#                         self.model.get_layer('encoder').trainable = True
#                         breakpoint()
#                 print(self.model.get_layer('encoder').trainable)

learning_rate = 0.001

def scheduler(epoch, lr):
        warmup_steps = 20

        if epoch < warmup_steps:
                new_lr =  learning_rate * min(1, (epoch+1)/warmup_steps)
                print('new_lr=', new_lr)
                return new_lr
        else:
                return lr

def callbacks(datetime_tag, model_name=None):

        log_dir = "./logs/"  'deepredmt/' + datetime_tag
        callbacks = [
                tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                               histogram_freq=1),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.1,
                                                     patience=3,
                                                     verbose=1),
                #CustomCallback(),
                # tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                #                                  patience=4,
                #                                  verbose=1),
                # tf.keras.callbacks.LearningRateScheduler(scheduler,
                #                                          verbose=0)
        ]

        # save the model with the best validation loss
        if model_name is not None:
                callbacks.append(
                        tf.keras.callbacks.ModelCheckpoint(
                                model_name,
                                save_best_only=True,
                                monitor='val_loss',
                                verbose=1)
                )

        return callbacks

def define_model(tf_fin):
        pretrained = tf.keras.models.load_model(tf_fin,
                                                compile=False)

        x = pretrained.input
        y = pretrained.output
        model = tf.keras.Model(inputs=x, outputs=y[1])

        return model

def tune(fin,
         tf_fin,
         augmentation=True,
         label_smoothing=True,
         num_hidden_units=5,
         batch_size=16,
         epochs=100,
         training_set_size=.8,
         save_model=False):

        train_gen, valid_gen = utils.prepare_dataset(
                fin,
                augmentation=True,
                label_smoothing=True,
                training_set_size=training_set_size,
                occlude_target=True,
                batch_size=batch_size
        )

        model = define_model(tf_fin)
        #model.get_layer('decoder').trainable = False
        model.summary()

        optimizer = tf.keras.optimizers.Adam()
        optimizer.learning_rate = 1e-5
        model.compile(optimizer,
                      loss = [
                          tf.keras.losses.CategoricalCrossentropy(),
                          tf.keras.losses.BinaryCrossentropy(),
                          tf.keras.losses.MeanAbsoluteError(),
                      ],
                      metrics = {
                          'cla': [
                              tf.keras.metrics.Precision(),
                              tf.keras.metrics.Recall(),
                              tf.keras.metrics.MeanSquaredError()
                          ],
                      },
                      #run_eagerly=True,
                      )

        datetime_tag = datetime.datetime.now().strftime("%y%m%d-%H%M")
        save_model = True
        if save_model:
                #model_name = "./models/" + 'deepredmt/' + datetime_tag + ".tf"
                model_name = './models/deepredmt/finetune-01.tf'
        else:
                model_name = None

        model_name = None
        model.fit(train_gen,
                  epochs=60,
                  validation_data=valid_gen,
                  callbacks=callbacks(datetime_tag, model_name),
                  workers=16)
