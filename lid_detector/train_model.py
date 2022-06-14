from __future__ import absolute_import
from __future__ import print_function

import argparse
from util import get_data, get_model, cross_entropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np


def train(batch_size=100, epochs=50):
    """
    Train one model with data augmentation: random padding+cropping and horizontal flip
    :param args: 
    :return: 
    """
    # print('Data set: %s' % dataset)
    X_train, Y_train, X_test, Y_test = get_data()
    model = get_model()
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.SGD(lr=0.05),
                  metrics=['accuracy'])

    benign_index_array = np.arange(X_train.shape[0])
    batch_num = np.int(np.ceil(X_train.shape[0] / batch_size))
    # training with data augmentation
    # data augmentation
    for i in np.arange(epochs):
        print("epoch {}".format(i))
        for j in np.arange(batch_num):
            x_batch = X_train[benign_index_array[
                              (j % batch_num) * batch_size:min((j % batch_num + 1) * batch_size, X_train.shape[0])], :]
            y_batch = Y_train[benign_index_array[
                              (j % batch_num) * batch_size:min((j % batch_num + 1) * batch_size, Y_train.shape[0])], :]
            model.train_on_batch(x_batch, y_batch)
        scores = model.evaluate(X_test, Y_test, verbose=1)
        print("Test loss: {}".format(scores[0]))
        print("Test accuracy: {}".format(scores[1]))

    model.save('data/model_att.h5')

def main(args):
    train(args.batch_size, args.epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--epochs',
        help="The number of epochs to train for.",
        required=False, type=int
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(epochs=50)
    parser.set_defaults(batch_size=100)
    args = parser.parse_args()
    main(args)
