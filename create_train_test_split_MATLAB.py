from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser
from utils import *
from scipy.io import savemat, loadmat
from sklearn.model_selection import train_test_split
import numpy as np
from config import Config


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', '-d',
                        help='Which split of the dataset to train/test the model on?' \
                             '(i.e. full, motorway, secondary or full_crossval)',
                        default='full_crossval')
    parser.add_argument('--preproc',
                        default='1')
    parser.add_argument('--input_dim',
                        help='Defines the input dimension of the VSA model' \
                             '(possible values are 512, 1024 or 2048)',
                        default=2048)
    parser.add_argument('--scale',
                        help='scaling of the scalar encoding with fractional binding ' \
                             '(possible values are 2, 4, 6, 8 and 10)',
                        default=6)

    args = parser.parse_args()

    config = Config()
    config.input_dim = int(args.input_dim)
    config.scale = float(args.scale)

    if args.preproc=='1':
        data = load_dataset(args.dataset,config)
        if type(data[0]) == list:
            X_train = []
            X_test = []
            y_train = []
            y_test = []
            for i in range(len(data[0])):
                config.n_time_measures = 1
                config.n_inputs = data[0][i].shape[2]
                config.n_steps = data[0][i].shape[1]
                t_train, X_train_, traces_train, init_vecs = create_HDC_vectors(config, data[0][i])
                t_test, X_test_, traces_test, init_vecs = create_HDC_vectors(config, data[1][i])

                # normalize HDC encodings
                m = np.mean(X_train_, axis=0)
                s = np.std(X_train_, axis=0)
                config.m = m
                config.s = s
                X_train_ = np.divide(X_train_ - m, s)
                X_test_ = np.divide(X_test_ - m, s)

                y_train_ = data[2][i]
                y_test_ = data[3][i]

                X_train.append(X_train_)
                X_test.append(X_test_)
                y_train.append(y_train_)
                y_test.append(y_test_)

        else:
            config.n_time_measures = 1
            config.n_inputs = data[0].shape[2]
            config.n_steps = data[0].shape[1]
            t_train, X_train, traces_train, init_vecs = create_HDC_vectors(config, data[0])
            t_test, X_test, traces_test, init_vecs = create_HDC_vectors(config, data[1])

            # normalize HDC encodings
            m = np.mean(X_train, axis=0)
            s = np.std(X_train, axis=0)
            config.m = m
            config.s = s
            X_train = np.divide(X_train - m, s)
            X_test = np.divide(X_test - m, s)

            y_train = data[2]
            y_test = data[3]
    else:
        data = load_dataset(args.dataset, config)

        if type(data[0]) == list:
            X_train = []
            X_test = []
            y_train = []
            y_test = []
            for i in range(len(data[0])):
                X_train_ = data[0][i]
                X_test_ = data[1][i]
                y_train_ = data[2][i]
                y_test_ = data[3][i]

                X_train.append(X_train_)
                X_test.append(X_test_)
                y_train.append(y_train_)
                y_test.append(y_test_)
        else:
            X_train = data[0]
            X_test = data[1]
            y_train = data[2]
            y_test = data[3]

    savemat('temp_data.mat',{'X_train':X_train, 'X_test':X_test,'Y_train':y_train,'Y_test':y_test})
