import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.io
import numpy as np


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape = shape)
  return tf.Variable(initial)

def activation(x):
    return tf.nn.relu(x)

def drop_out(x):
    return tf.nn.dropout(x, 0.7)

def inference(x):
    with tf.variable_scope('dense1'):
        W_1 = weight_variable([10000,30000])
        b_1 = bias_variable([30000])
        x = tf.matmul(x, W_1) + b_1
        x = activation(x)
    with tf.variable_scope('dropout1'):
        x = drop_out(x)
    with tf.variable_scope('dense2'):
        W_2 = weight_variable([30000,25])
        b_2 = bias_variable([25])
        print(x.shape)
        output = tf.matmul(x, W_2) + b_2
    return output


class NiN_Model():
    def __init__(self):
        self.x_input = tf.placeholder(tf.float32, shape=[None, 10000])
        self.y_input = tf.placeholder(tf.float32, shape=[None, 25])
        self.bs = tf.placeholder(tf.int32, shape=None)

        self.y = inference(self.x_input)
        self.predictions = tf.argmax(self.y, 1)
        self.y_input_ = tf.argmax(self.y_input, 1)
        self.correct_prediction = tf.equal(self.predictions, self.y_input_)

        self.corr_pred = self.correct_prediction

        self.y_xent = tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_input)
        self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
        self.grad = tf.gradients(self.xent, self.x_input)[0]

        self.num_correct = tf.reduce_sum(tf.cast(self.correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))