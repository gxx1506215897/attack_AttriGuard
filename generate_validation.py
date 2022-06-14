import torch
import torchvision
import numpy as np
from input_data import *

t=Data_Class()
t.input_train_app()
t.input_train_label()
t.input_test_app()
t.input_test_label()

# input_shape=t.train_app.shape[1:]
x_train=t.train_app
y_train=t.train_label
x_test=t.test_app
y_test=t.test_label


np.random.seed(0)
m = 14607
P = np.random.permutation(m)
n = 1000

val_data = x_train[P[:n]]
val_labels = y_train[P[:n]]
train_data = x_train[P[n:]]
train_labels = y_train[P[n:]]
test_data = x_test
test_label = y_test