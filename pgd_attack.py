"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np



class L0PGDAttack:
  def __init__(self, model, epsilon, num_iter, k, a, random_start, loss_func, lb, ub):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.k = k
    self.a = a
    self.rand = random_start
    self.num_iter = num_iter
    self.lb = lb
    self.ub = ub

    if loss_func == 'xent':
      loss = model.xent
    elif loss_func == 'cw':
      label_mask = model.y_input
      correct_logit = tf.reduce_sum(label_mask * model.y, axis=1) #每一行求和。
      wrong_logit = tf.reduce_max((1-label_mask) * model.y - 1e4*label_mask, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]
  def clip_att(self, a):
    b1 = a <= 0.1
    b2 = (a > 0.1) & (a <= 0.3)
    b3 = (a > 0.3) & (a <= 0.5)
    b4 = (a > 0.5) & (a <= 0.7)
    b5 = (a > 0.7) & (a <= 0.9)
    b6 = (a > 0.9)
    a1 = 0.0*b1
    a2 = 0.2*b2
    a3 = 0.4*b3
    a4 = 0.6*b4
    a5 = 0.8*b5
    a6 = 1.0*b6
    c = a1 + a2 + a3 + a4 + a5 + a6
    return c

  def project_L0_box(self, y, k, lb, ub):
    ''' projection of the batch y to a batch x such that:
          - each image of the batch x has at most k pixels with non-zero channels
          - lb <= x <= ub '''
    x = np.copy(y)
    p1 = x ** 2
    p2 = np.minimum(np.minimum(ub - x, x - lb), 0)
    p2 = p2 ** 2
    p3 = (np.sort(p1 - p2, axis=1)[:, -k]).reshape(x.shape[0], -1)
    x = x * (np.logical_and(lb <= x, x <= ub)) + lb * (lb > x) + ub * (x > ub)
    x *= ((p1 - p2) >= p3)
    return x

  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      x = np.clip(x, 0, 1) # ensure valid pixel range
    else:
      x = np.copy(x_nat)

    for i in range(self.num_iter):
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      grad /= (1e-10 + np.sum(np.abs(grad), axis=1, keepdims=True))
      x = np.add(x, (np.random.random_sample(grad.shape)-0.5)*1e-12 + self.a * grad, casting='unsafe')
      # x = self.clip_att(x_nat + self.project_L0_box(x - x_nat, self.k, self.lb, self.ub))
      x = x_nat + self.project_L0_box(x - x_nat, self.k, self.lb, self.ub)

    return x


# if __name__ == '__main__':
#   import json
#   import sys
#   import math

#   from tensorflow.examples.tutorials.mnist import input_data

#   from model import Model

#   with open('config.json') as config_file:
#     config = json.load(config_file)

#   model_file = tf.train.latest_checkpoint(config['model_dir'])
#   if model_file is None:
#     print('No model found')
#     sys.exit()

#   model = Model()
#   attack = LinfPGDAttack(model,
#                          config['epsilon'],
#                          config['k'],
#                          config['a'],
#                          config['random_start'],
#                          config['loss_func'])
#   saver = tf.train.Saver()

#   mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

#   with tf.Session() as sess:
#     # Restore the checkpoint
#     saver.restore(sess, model_file)

#     # Iterate over the samples batch-by-batch
#     num_eval_examples = config['num_eval_examples']
#     eval_batch_size = config['eval_batch_size']
#     num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

#     x_adv = [] # adv accumulator

#     print('Iterating over {} batches'.format(num_batches))

#     for ibatch in range(num_batches):
#       bstart = ibatch * eval_batch_size
#       bend = min(bstart + eval_batch_size, num_eval_examples)
#       print('batch size: {}'.format(bend - bstart))

#       x_batch = mnist.test.images[bstart:bend, :]
#       y_batch = mnist.test.labels[bstart:bend]

#       x_batch_adv = attack.perturb(x_batch, y_batch, sess)

#       x_adv.append(x_batch_adv)

#     print('Storing examples')
#     path = config['store_adv_path']
#     x_adv = np.concatenate(x_adv, axis=0)
#     np.save(path, x_adv)
#     print('Examples stored in {}'.format(path))
