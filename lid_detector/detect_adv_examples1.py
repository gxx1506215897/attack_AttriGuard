from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import numpy as np
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from util import (random_split, block_split, train_lr, compute_roc)

DATASETS = ['mnist', 'cifar', 'svhn','att']
ATTACKS = ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2','attriguard']
CHARACTERISTICS = ['kd', 'bu', 'lid']
PATH_DATA = "data/"
PATH_IMAGES = "plots/"
PATH_DATA1 = './data_grid_search/lid_large_batch/'
PATH_DATA2 = './data_grid_search/attriguard_adv_lid/'

def load_characteristics(dataset, attack, characteristics, file_name):
    """
    Load multiple characteristics for one dataset and one attack.
    :param dataset: 
    :param attack: 
    :param characteristics: 
    :return: 
    """
    X, Y = None, None
    for characteristic in characteristics:
        # print("  -- %s" % characteristics)
        data = np.load(file_name)
        if X is None:
            X = data[:, :-1]
        else:
            X = np.concatenate((X, data[:, :-1]), axis=1)
        if Y is None:
            Y = data[:, -1] # labels only need to load once

    return X, Y

def detect(args):
    assert args.dataset in DATASETS, \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.attack in ATTACKS, \
        "Train attack must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma', 'cw-l2'"
    assert args.test_attack in ATTACKS, \
        "Test attack must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma', 'cw-l2'"
    characteristics = args.characteristics.split(',')
    for char in characteristics:
        assert char in CHARACTERISTICS, \
            "Characteristic(s) to use 'kd', 'bu', 'lid'"

    print("Loading train attack: %s" % args.attack)
    print(characteristics[0])
    train_path = os.path.join(PATH_DATA1, "%s_%s_%s_20.npy" % (characteristics[0], args.dataset, args.attack))
    X, Y = load_characteristics(args.dataset, args.attack,characteristics, train_path)

    # standarization
    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)
    # X = scale(X) # Z-norm
    tj = np.zeros([25,5])
    for i in range(25):
        tj[i,0] = i
    # test attack is the same as training attack
        X_train, Y_train, X_test, Y_test = block_split(X, Y)
        lr = train_lr(X_train, Y_train)
        if args.test_attack != args.attack:
            # test attack is a different attack
            print("Loading test attack: %s" % args.test_attack)
            test_path = os.path.join(PATH_DATA2,str(i)+'.npy')
            X_test, Y_test = load_characteristics(args.dataset, args.test_attack, characteristics, test_path)
            _, _, X_test, Y_test = block_split(X_test, Y_test)

            # apply training normalizer
            X_test = scaler.transform(X_test)
            # X_test = scale(X_test) # Z-norm
        
        print("Train data size: ", X_train.shape)
        print("Test data size: ", X_test.shape)


        ## Build detector
        print("LR Detector on [dataset: %s, train_attack: %s, test_attack: %s] with:" %
                                            (args.dataset, args.attack, args.test_attack))

        ## Evaluate detector
        y_pred = lr.predict_proba(X_test)[:, 1]
        y_label_pred = lr.predict(X_test)

        # AUC
        _, _, auc_score = compute_roc(Y_test, y_pred, plot=False)
        precision = precision_score(Y_test, y_label_pred)
        recall = recall_score(Y_test, y_label_pred)

        y_label_pred = lr.predict(X_test)
        acc = accuracy_score(Y_test, y_label_pred)
        print('Detector ROC-AUC score: %0.4f, accuracy: %.4f, precision: %.4f, recall: %.4f' % (auc_score, acc, precision, recall))
        tj[i,1], tj[i,2],tj[i,3], tj[i,4] = auc_score, acc, precision, recall
    np.savetxt('tongji2.csv', tj, delimiter = ',') 
    return lr, auc_score, scaler
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use train the discriminator; either 'fgsm', 'bim-a', 'bim-b', 'jsma' 'cw-l2'",
        required=True, type=str
    )
    parser.add_argument(
        '-r', '--characteristics',
        help="Characteristic(s) to use any combination in ['kd', 'bu', 'lid'] "
             "separated by comma, for example: kd,bu",
        required=True, type=str
    )
    parser.add_argument(
        '-t', '--test_attack',
        help="Characteristic(s) to cross-test the discriminator.",
        required=False, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(batch_size=100)
    parser.set_defaults(test_attack=None)
    args = parser.parse_args()
    detect(args)
