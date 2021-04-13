#!/usr/bin/env python3
import sys
import math
import datetime
import tempfile

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
from . import data_handler
from . import models
from . import losses
from . import metrics
from . import layers

# define output model name
datetime_tag = datetime.datetime.now().strftime('%y%m%d%H%M')
model_fout = tempfile.gettempdir() + '/' + datetime_tag + '_deepredmt.tf'

num_classes = 2 # edited and unedited, two classes

loss_avg = tf.keras.metrics.Mean()
metric_avg = {'sen': tf.keras.metrics.Mean(),
              'pre': tf.keras.metrics.Mean()}



@tf.function
def calculate_metrics(y_true, y_pred):
        # calculate performance metrics independently for each label
        return {
                'sen': metrics._sen_fn(y_true, y_pred),
                'pre': metrics._pre_fn(y_true, y_pred),
        }

@tf.function
def train_step(ds, model, opt):
        x_occluded, y_true, ext_true, x_no_occluded = ds

        with tf.GradientTape() as tape:
                out = model(x_occluded)
                x_recon = out[0]
                y_pred = out[2]
                # batch_reconstruction does not have occlusion
                recon_loss = losses.reconstruction_loss_fn(x_no_occluded, x_recon)
                class_loss = losses.classification_loss_fn(y_true,
                                                           y_pred,
                                                           ext_true,
                                                           _label_smoothing)
                regularization = losses.regularization(model.trainable_variables)
                loss = recon_loss + class_loss + regularization
        grads = tape.gradient(loss, model.trainable_variables)
        # apply gradient to main model
        opt.apply_gradients(zip(grads, model.trainable_variables))

        metrics = calculate_metrics(y_true, y_pred)

        return [loss, metrics]

@tf.function
def test_step(ds, model, opt):
        x, y_true, ext_true, _ = ds

        out = model(x, training=False)
        x_recon = out[0]
        y_pred = out[2]
        recon_loss = losses.reconstruction_loss_fn(x, x_recon)
        class_loss = losses.classification_loss_fn(y_true,
                                                   y_pred,
                                                   ext_true,
                                                   _label_smoothing)
        regularization = losses.regularization(model.trainable_variables)
        loss = recon_loss + class_loss + regularization

        metrics = calculate_metrics(y_true, y_pred)

        return [loss, metrics]

def print_minibatch_progress(step, data_gen):
        BAR = [ '-', '\\', '|', '/' ]
        print("Epoch %d %s training step %d/%d" % (epoch_counter, BAR[step % 4], step, len(data_gen)), end='\r')

def epoch_step(data_gen, mini_batch_step_fn, model, opt, print_log=True):
        # reset values
        loss_avg.reset_states()
        [metric_avg[k].reset_states for k in metric_avg]

        # performing over mini batches
        for step in range(len(data_gen)):
                if print_log:
                        print_minibatch_progress(step, data_gen)
                # update parameters
                loss, metr = mini_batch_step_fn(data_gen[step], model, opt)

                # Track progress
                loss_avg(loss)
                # performance metrics
                [metric_avg[k](metr[k].numpy()) for k in metr]

        loss = loss_avg.result().numpy()
        metric_results = [metric_avg[k].result().numpy() for k in metric_avg]

        return loss, metric_results

def print_epoch_progress(loss,
                         vl_loss,
                         metr,
                         vl_metr):
        print('Epoch %d Tr %.4f Vl %.4f' % (epoch_counter, loss[-1], vl_loss[-1]))

        outstr = 'sen %.4f pre %.4f val_sen %.4f val_pre %.4f\n' % (metr[0], metr[1], vl_metr[0], vl_metr[1])
        print(outstr.strip())

def fit(train_gen,
        valid_gen,
        num_hidden_units=5,
        label_smoothing=True,
        epochs=100):
        global _label_smoothing
        _label_smoothing = label_smoothing

        # define model and optimizer
        model = models.CAE(input_shape=(41, 4),
                           num_hunits=num_hidden_units)
        opt = tf.keras.optimizers.Adam(learning_rate=0.001,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       amsgrad=False)

        ## model optimization
        tr_losses = []
        vl_losses = []

        best_valid_loss = np.inf
        patience = 3 # number of epochs with no improvement
        patience_cnt = 0
        lr_decay = 0.1

        global epoch_counter
        epoch_counter = 0
        global decay_counter
        decay_counter = 1
        max_num_attempts = 3 # maximun number of LR decays until stopping learning

        for epoch_counter in range(1, epochs):
                # train phase
                loss, metrs = epoch_step(train_gen, train_step, model, opt, True)
                tr_losses.append(loss)

                # validation step
                val_loss, val_metrs = epoch_step(valid_gen, test_step, model, opt, False)
                vl_losses.append(val_loss)

                # stop learning if there are nan values
                if math.isnan(tr_losses[-1]) or math.isnan(vl_losses[-1]):
                        print('Stopping learning. Nan values found.')
                        break

                # one epoch end
                if val_loss <= best_valid_loss:
                        checkpoint_str = '* ---> Saving model as %.4f is less than %.4f\n' % (val_loss, best_valid_loss)
                        best_valid_loss = val_loss
                        patience_cnt = 0
                        model.save(model_fout)
                        print(checkpoint_str)
                if val_loss > best_valid_loss:
                        patience_cnt += 1

                        if patience_cnt >= patience:
                                if decay_counter > max_num_attempts:
                                        break
                                else:
                                        decay_counter +=1
                                        patience_cnt = 0
                                        opt.lr = opt.lr * lr_decay
                                        print("Learning rate reduced\n")
                # show progress
                if True : #epoch_counter % 10 == 0:
                        print_epoch_progress(tr_losses, vl_losses, metrs, val_metrs)
                # shuffle generators
                train_gen.on_epoch_end()
                valid_gen.on_epoch_end()

        # load model with best performance on the valid set
        model = tf.keras.models.load_model(model_fout, compile=False)

        return model
