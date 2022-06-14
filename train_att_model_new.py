"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from input_data import *
from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer
import numpy as np
from att_NiN_bn import NiN_Model
from pgd_attack import *
import time
from generate_validation import *


def next_batch(train_data, train_target, batch_size):
    #打乱数据集
    index = [ i for i in range(0,train_target.shape[0]) ]
    np.random.shuffle(index)
    #建立batch_data与batch_target的空列表
    batch_data = []
    batch_target = []
    #向空列表加入训练集及标签
    for i in range(0,batch_size):
        batch_data.append(train_data[index[i]])
        batch_target.append(train_target[index[i]])
    return np.array(batch_data), np.array(batch_target)

# from pgd_attack import LinfPGDAttack

with open('config_new.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.random.set_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

mode = config['mode']

batch_size = config['training_batch_size']

# Setting up the data and the model
global_step = tf.compat.v1.train.get_or_create_global_step()
model = NiN_Model()

# Setting up the optimizer
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.001).minimize(model.xent, global_step=global_step)
# train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(model.xent, global_step=global_step)


# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.compat.v1.train.Saver(max_to_keep=3)

shutil.copy('config_new.json', model_dir)
if mode == "NAT":
    with tf.compat.v1.Session() as sess:
        # Initialize the summary writer, global variables, and our time counter.
        summary_writer = tf.compat.v1.summary.FileWriter(model_dir, sess.graph)
        sess.run(tf.compat.v1.global_variables_initializer())
        training_time = 0.0
        start0 = time.time()
        # Main training loop
        for ii in range(max_num_training_steps):
            x_batch, y_batch = next_batch(train_data, train_labels, batch_size)

            # Compute Adversarial Perturbations
            start = timer()
            # x_batch_adv = attack.perturb(x_batch, y_batch, sess)
            end = timer()
            training_time += end - start

            nat_dict = {model.x_input: x_batch,
                        model.y_input: y_batch}
                        # Actual training step

            if ii % num_output_steps == 0:
                nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
                print('Step {}:    ({})'.format(ii, datetime.now()))
                print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
            if ii != 0:
                print('    {} examples per second'.format(num_output_steps * batch_size /training_time))
                training_time = 0.0

            if ii % num_checkpoint_steps == 0:
                saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)
            start = timer()
            sess.run(train_step, feed_dict=nat_dict)
            end = timer()
            training_time += end - start
        end0 = time.time()
        print(end0-start0)

elif mode == "ROB":
    # Set up adversary
    attack = L0PGDAttack(model,
                       config['epsilon'],
                       config['num_iter'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'],
                       config['lb'],
                       config['ub'])
    tf.compat.v1.summary.scalar('accuracy adv train', model.accuracy)
    tf.compat.v1.summary.scalar('accuracy adv', model.accuracy)
    tf.compat.v1.summary.scalar('xent adv train', model.xent / batch_size)
    tf.compat.v1.summary.scalar('xent adv', model.xent / batch_size)
    # tf.summary.image('images adv train', model.x_image)
    merged_summaries = tf.compat.v1.summary.merge_all()
    with tf.compat.v1.Session() as sess:
        # Initialize the summary writer, global variables, and our time counter.
        summary_writer = tf.compat.v1.summary.FileWriter(model_dir, sess.graph)
        sess.run(tf.compat.v1.global_variables_initializer())
        best_acc_val = 0
        training_time = 0.0
        start0 = time.time()
        # Main training loop
        for ii in range(max_num_training_steps):
            x_batch, y_batch = next_batch(x_train, y_train, batch_size)

            # Compute Adversarial Perturbations
            start = timer()
            x_batch_adv = attack.perturb(x_batch, y_batch, sess)
            end = timer()
            training_time += end - start

            nat_dict = {model.x_input: x_batch,
                        model.y_input: y_batch}

            adv_dict = {model.x_input: x_batch_adv,
                        model.y_input: y_batch}

            # Output to stdout
            if ii % num_output_steps == 0:
                nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
                adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
                print('Step {}:    ({})'.format(ii, datetime.now()))
                print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
                print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
                if ii != 0:
                    print('    {} examples per second'.format(num_output_steps * batch_size / training_time))
                    training_time = 0.0
            # Tensorboard summaries
            if ii % num_summary_steps == 0:
                summary = sess.run(merged_summaries, feed_dict=adv_dict)
                summary_writer.add_summary(summary, global_step.eval(sess))
            sess.run(train_step, feed_dict=adv_dict)
            val_index_array = np.arange(val_data.shape)
            np.random.shuffle(val_index_array)
            val_batch_num = np.int(np.ceil(val_data.shape[0]/batch_size))
            num_correct_val = 0
            for jj in range(val_batch_num):
                x_batch_val = val_data[val_index_array[(jj % val_batch_num) * batch_size:min((jj % val_batch_num + 1) * batch_size, val_data.shape[0])],:]
                y_batch_val = val_labels[val_index_array[(jj % val_batch_num) * batch_size:min((jj % val_batch_num + 1) * batch_size, val_data.shape[0])],:]
                x_batch_val_adv = attack.perturb(x_batch_val,y_batch_val,sess)
                adv_dict_val = {model.x_input: x_batch_val_adv, model.y_input: y_batch_val}
                num_correct_val += sess.run(model.num_correct, feed_dict=adv_dict_val)
            acc_val = (num_correct_val*1.0)/val_data.shape[0]
            if acc_val >best_acc_val:
                best_acc_val = acc_val
                saver.save(sess, os.path.join(model_dir, 'checkpoint'),global_step=global_step)
        end0 = time.time()
        print(end0-start0)
