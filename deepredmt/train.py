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
import numpy as np
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# define output model name
datetime_tag = datetime.datetime.now().strftime('%y%m%d%H%M')
model_fout = tempfile.gettempdir() + '/' + datetime_tag + '_deepredmt.tf'

num_classes = 2 # edited and unedited, two classes

from . import utils
from . import data_handler
from . import models
from . import losses
from . import metrics
from . import layers

@tf.function
def calculate_metrics(class_true, class_pred):
        # calculate performance metrics independently for each label
        metric_values = []
        # parse class prediction
        max_classes = tf.math.argmax(class_pred, axis=1)
        parsed_class_pred = tf.one_hot(max_classes,
                                       num_classes=num_classes,
                                       dtype='float32')
        for i in range(num_classes):
                c_t = class_true[:, i]
                c_p = parsed_class_pred[:, i]
                sen = metrics._sen_fn(c_t, c_p) # sensitivity
                pre = metrics._pre_fn(c_t, c_p) # precision
                metric_values.append({'sen':sen, 'pre':pre})
        return metric_values

@tf.function
def train_step(ds, model, opt):
        batch, class_true, ext_true, batch_reconstruction = ds

        with tf.GradientTape() as tape:
                out = model(batch)
                recon = out[0]
                class_pred = out[2]
                # batch_reconstruction does not have occlusion
                recon_loss = losses.reconstruction_loss_fn(batch_reconstruction, recon)
                class_loss = losses.classification_loss_fn(class_true,
                                                           class_pred,
                                                           ext_true,
                                                           _label_smoothing)
                regularization = losses.regularization(model.trainable_variables)
                loss = recon_loss + class_loss + regularization
        grads = tape.gradient(loss, model.trainable_variables)
        # apply gradient to main model
        opt.apply_gradients(zip(grads, model.trainable_variables))

        del tape  # Drop reference to the tape

        metrics = calculate_metrics(class_true, class_pred)

        return [loss, metrics]

@tf.function
def test_step(ds, model, opt):
        batch, class_true, ext_true, batch_reconstruction = ds

        out = model.predict(batch)
        recon = out[0]
        class_pred = out[2]
        recon_loss = losses.reconstruction_loss_fn(batch, recon)
        class_loss = losses.classification_loss_fn(class_true,
                                                   class_pred,
                                                   ext_true,
                                                   _label_smoothing)
        regularization = losses.regularization(model.trainable_variables)
        loss = recon_loss + class_loss + regularization

        metrics = calculate_metrics(class_true, class_pred)

        return [loss, metrics]

def print_minibatch_progress(step, data_gen):
        BAR = [ '-', '\\', '|', '/' ]
        print("Epoch %d %s training step %d/%d" % (epoch_counter, BAR[step % 4], step, len(data_gen)), end='\r')

def epoch_step(data_gen, mini_batch_step_fn, model, opt, print_log=True):
        loss_avg = tf.keras.metrics.Mean()

        metric_avg = [{'sen': tf.keras.metrics.Mean(),
                       'pre': tf.keras.metrics.Mean()} for i in range(num_classes)]
        # performing over mini batches
        for step in range(len(data_gen)):
                if print_log:
                        print_minibatch_progress(step, data_gen)
                # update parameters
                loss, metr = mini_batch_step_fn(data_gen[step], model, opt)
                # Track progress
                loss_avg(loss)

                # performance metrics
                for i in range(num_classes):
                        for k in metric_avg[i]:
                                metric_avg[i][k](metr[i][k])
        loss = float(loss_avg.result())

        metric_results = []
        for i in range(num_classes):
                metric_results.append({})
                for metric in metric_avg[i]:
                        metric_results[i][metric] = float(metric_avg[i][metric].result())
        return [loss, metric_results]

def print_epoch_progress(tr_loss,
                         vl_loss,
                         tr_metr,
                         vl_metr):
        print('Epoch %d Tr %.4f Vl %.4f' % (epoch_counter, tr_loss[-1], vl_loss[-1]))
        outstr = ''
        for i in range(num_classes):
                outstr += 'tr_sen[%d] %.4f vl_sen[%d] %.4f\n' % (i, tr_metr[i]['sen'][-1], i, vl_metr[i]['sen'][-1])
        print(outstr.strip())

def fit(train_gen,
        valid_gen,
        num_hidden_units=5,
        data_augmentation=True,
        label_smoothing=True,
        tr_log_fout=None,
        vl_log_fout=None):
        global _label_smoothing
        _label_smoothing = label_smoothing

        datetime_tag = datetime.datetime.now().strftime('%y%m%d%H%M')
        if tr_log_fout is None:
                tr_log_fout = tempfile.gettempdir() + '/' + datetime_tag + '.tr.log'
        if vl_log_fout is None:
                vl_log_fout = tempfile.gettempdir() + '/' + datetime_tag + '.vl.log'

        tr_log_fd = open(tr_log_fout, 'a')
        vl_log_fd = open(vl_log_fout, 'a')

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
        # test_losses = []
        tr_metr = [{'sen':[], 'pre':[]} for i in range(num_classes)]
        vl_metr = [{'sen':[], 'pre':[]} for i in range(num_classes)]
        best_valid_loss = float(math.inf)
        patience = 3 # number of epochs with no improvement
        patience_cnt = 0
        checkpointing_string = ''
        lr_decay = 0.1

        global epoch_counter
        epoch_counter = 0
        global decay_counter
        decay_counter = 1
        max_num_attempts = 3 # maximun number of LR decays until stopping learning
        while(True):
                epoch_counter += 1
                # train phase
                train_loss, metrs = epoch_step(train_gen, train_step, model, opt, True)
                tr_losses.append(train_loss)
                # save metrics
                for i in range(num_classes):
                        for k in ('sen', 'pre'):
                                tr_metr[i][k].append(metrs[i][k])
                # testing
                valid_loss, metrs = epoch_step(valid_gen, test_step, model, opt, False)
                vl_losses.append(valid_loss)
                # save metrics
                for i in range(num_classes):
                        for k in ('sen', 'pre'):
                                vl_metr[i][k].append(metrs[i][k])
                # stop learning if there are nan values
                if math.isnan(tr_losses[-1]) or math.isnan(vl_losses[-1]):
                        print('Stopping learning. Nan values found.')
                        break
                # one epoch end
                if valid_loss <= best_valid_loss:
                        checkpoint_str = '* ---> Saving model as %.4f is less than %.4f\n' % (valid_loss, best_valid_loss)
                        for i in range(num_classes):
                                checkpoint_str += '  tr_sen[%d] %.4f vl_sen[%d] %.4f\n' % (i, tr_metr[i]['sen'][-1], i, vl_metr[i]['sen'][-1])
                        print(checkpoint_str.strip())
                        best_valid_loss = valid_loss
                        model.save(model_fout)
                        patience_cnt = 0
                if valid_loss > best_valid_loss:
                        patience_cnt += 1
                        #if patience_cnt >= tf.math.ceil(patience/decay_counter):
                        if patience_cnt >= patience:
                                print("Reducing learning rate")
                                if decay_counter > max_num_attempts:
                                        break
                                else:
                                        decay_counter +=1
                                        patience_cnt = 0
                                        opt.lr = opt.lr * lr_decay
                # show progress
                if True : #epoch_counter % 10 == 0:
                        print_epoch_progress(tr_losses, vl_losses, tr_metr, vl_metr)
                # shuffle generators
                train_gen.on_epoch_end()
                valid_gen.on_epoch_end()

                # save loss history
                outvalues = [tr_losses[-1]]
                for i in range(num_classes):
                        outvalues.append(tr_metr[i]['sen'][-1])
                tr_log_fd.write('\t'.join(map(str, outvalues)) + '\n')
                tr_log_fd.flush()
                outvalues = [vl_losses[-1]]
                for i in range(num_classes):
                        outvalues.append(vl_metr[i]['sen'][-1])
                vl_log_fd.write('\t'.join(map(str, outvalues)) + '\n')
                vl_log_fd.flush()
        tr_log_fd.close()
        vl_log_fd.close()

        # return model with best performance on the valid set
        model = tf.keras.models.load_model(model_fout, compile=False)

        return model
