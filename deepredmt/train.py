#!/usr/bin/env python3
import sys
import numpy as np
import math

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

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

from src import utils
from src import data_handler
from src import models
from src import losses
from src import metrics
from src import layers
from src import losses

def train_step(ds, model, opt):
        return basic_step(ds, model, opt, True)

def test_step(ds, model, opt):
        return basic_step(ds, model, None, False)

def basic_step(ds, M, opt, train_flag):
        batch, class_true, ext_true, batch_reconstruction = ds
        # calculate loss and derivatives
        if train_flag:
                #with tf.GradientTape(persistent=True) as tape:
                with tf.GradientTape() as tape:
                        out = M(batch)
                        recon = out[0]
                        class_pred = out[2]
                        # batch_reconstruction does not have occlusion
                        recon_loss = losses.reconstruction_loss_fn(batch_reconstruction, recon)
                        class_loss = losses.classification_loss_fn(class_true,
                                                                   class_pred,
                                                                   ext_true,
                                                                   _label_smoothing)
                        regularization = losses.regularization(M.trainable_variables)
                        loss = recon_loss + class_loss + regularization
                        grads = tape.gradient(loss, M.trainable_variables)
                # apply gradient to main model
                opt.apply_gradients(zip(grads, M.trainable_variables))
                norm = tf.linalg.global_norm(grads)
                pcorrect = 0.0 # don't trace percentage of correct predictions
                del tape  # Drop reference to the tape
        else:
                out = M(batch)
                recon = out[0]
                class_pred = out[2]
                recon_loss = losses.reconstruction_loss_fn(batch, recon)
                class_loss = losses.classification_loss_fn(class_true,
                                                           class_pred,
                                                           ext_true,
                                                           _label_smoothing)
                regularization = losses.regularization(M.trainable_variables)
                loss = recon_loss + class_loss + regularization
                norm = 0.0
                pcorrect = tf.reduce_mean(tf.reduce_sum(class_true * class_pred, axis=1))
        # calculate performance metrics independently for each label
        metric_values = []
        # parse class prediction
        parsed_class_pred = keras.utils.to_categorical(tf.math.argmax(class_pred, axis=1),
                                                       num_classes=num_classes,
                                                       dtype='float32')
        for i in range(num_classes):
                c_t = class_true[:, i]
                c_p = parsed_class_pred[:, i]
                sen = metrics._sen_fn(c_t, c_p) # sensitivity
                pre = metrics._pre_fn(c_t, c_p) # precision
                metric_values.append({'sen':sen, 'pre':pre})
        return [loss, norm, metric_values, pcorrect]#, cp_performance]

def print_minibatch_progress(step, data_gen):
        BAR = [ '-', '\\', '|', '/' ]
        print("Epoch %d %s training step %d/%d" % (epoch_counter, BAR[step % 4], step, len(data_gen)), end='\r')

def epoch_step(data_gen, mini_batch_step_fn, model, opt, print_log=True):
        loss_avg = tf.keras.metrics.Mean()
        norm_avg = tf.keras.metrics.Mean()
        metric_avg = [{'sen': tf.keras.metrics.Mean(), 'pre': tf.keras.metrics.Mean()} for i in range(num_classes)]
        pcor_avg =  tf.keras.metrics.Mean()
        # performing over mini batches
        for step in range(len(data_gen)):
                if print_log:
                        print_minibatch_progress(step, data_gen)
                # update parameters
                loss, norm, metr, pcorrect = mini_batch_step_fn(data_gen[step], model, opt)
                # Track progress
                loss_avg(loss)
                norm_avg(norm)
                # performance metrics
                for i in range(num_classes):
                        for k in metric_avg[i]:
                                metric_avg[i][k](metr[i][k])
                # accuracy
                pcor_avg(pcorrect)
        loss = float(loss_avg.result())
        norm = float(norm_avg.result())
        metric_results = []
        for i in range(num_classes):
                metric_results.append({})
                for metric in metric_avg[i]:
                        metric_results[i][metric] = float(metric_avg[i][metric].result())
        pcor_avg = float(pcor_avg.result())
        return [loss, norm, metric_results, pcorrect]

def print_epoch_progress(norm, tr_loss, vl_loss, tr_metr, vl_metr, pcor, rpcor):
        print('Epoch %d Norm %.4f Tr %.4f Vl %.4f PCOR: %.4f RPCOR: %.4f' % (epoch_counter, norm, tr_loss[-1], vl_loss[-1], pcor, rpcor))
        outstr = ''
        for i in range(num_classes):
                outstr += 'tr_sen[%d] %.4f vl_sen[%d] %.4f\n' % (i, tr_metr[i]['sen'][-1], i, vl_metr[i]['sen'][-1])
        print(outstr.strip())

def train_model(num_hidden_units,
                data_augmentation,
                label_smoothing,
                win_fin,
                model_fout,
                tr_log_fout,
                vl_log_fout):
        global _label_smoothing
        _label_smoothing = label_smoothing
        tr_log_fd = open(tr_log_fout, 'a')
        vl_log_fd = open(vl_log_fout, 'a')
        # read windows
        # (label, win, ext, cp, key)
        data  = data_handler.read_windows(win_fin)
        global num_classes
        num_classes = len(set(data[:, 0]))
        # generate a train and valid set for each label
        neg_train_idx, neg_valid_idx = data_handler.train_test_split(np.where(data[:,0] == 0)[0])
        pos_train_idx, pos_valid_idx = data_handler.train_test_split(np.where(data[:,0] == 1)[0])
        train_idx = neg_train_idx + pos_train_idx
        valid_idx = neg_valid_idx + pos_valid_idx
        train_set = np.array([ data[i] for i in train_idx ])
        valid_set = np.array([ data[i] for i in valid_idx ])
        train_gen = utils.DataGenerator(train_set,
                                        batch_size=16,
                                        data_augmentation=data_augmentation,
                                        occlusion=data_augmentation,
                                        shuffle=True)
        valid_gen = utils.DataGenerator(valid_set,
                                        batch_size=16,
                                        data_augmentation=False,
                                        occlusion=False,
                                        shuffle=True)
        # model
        model = models.CAE(input_shape=(41, 4), num_hunits=num_hidden_units, filters=[16, 32, 64, 128, 256, 512])
        opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

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
        running_pcor = 0.5
        global epoch_counter
        epoch_counter = 0
        global decay_counter
        decay_counter = 1
        max_num_attempts = 3 # maximun number of LR decays until stopping learning
        while(True):
                epoch_counter += 1
                # train phase
                train_loss, norm, metrs, dummy_pcor = epoch_step(train_gen, train_step, model, opt, True)
                tr_losses.append(train_loss)
                # save metrics
                for i in range(num_classes):
                        for k in ('sen', 'pre'):
                                tr_metr[i][k].append(metrs[i][k])
                # testing
                valid_loss, dummy_norm, metrs, pcor = epoch_step(valid_gen, test_step, model, opt, False)
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
                        print_epoch_progress(norm, tr_losses, vl_losses, tr_metr, vl_metr, pcor, running_pcor)
                # shuffle generators
                train_gen.on_epoch_end()
                valid_gen.on_epoch_end()
                running_pcor = running_pcor * .9 + float(pcor) * .1
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
