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
                        default='driving_style')
    parser.add_argument('--preproc',
                        default='0')
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
    config.input_dim = args.input_dim
    config.scale = args.scale

    if args.preproc=='1':
        data = load_dataset(args.dataset,config,True)
    else:
        data = load_dataset(args.dataset, config, False)
    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]

    savemat('temp_data.mat',{'X_train':X_train, 'X_test':X_test,'Y_train':y_train,'Y_test':y_test})
