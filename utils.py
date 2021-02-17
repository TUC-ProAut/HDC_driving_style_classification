import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os
import cmath
import tensorflow as tf
from joblib import Parallel, delayed
import multiprocessing
import io
from scipy.io import savemat, loadmat
from keras.callbacks import Callback
from time import time
from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
    
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 

    return batch_s

def one_hot(y_,n_classes=-1):
    # Function to encode output labels from number indexes 
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    if n_classes==-1:
        n_values = np.max(y_) + 1
    else:
        n_values = n_classes
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

def load_motorway_dataset(data_path='data'):
    # Function to load the motorway dataset only 

    with open(os.path.join(data_path, 'motorway_dataset_window_64_proc_veh_DtA.pkl'), 'rb') as f:
        save = pickle.load(f, encoding='latin1')
        motorway_dataset = save['dataset']
        motorway_labels = save['labels']
        del save
        print('Motorway set', motorway_dataset.shape, motorway_labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(motorway_dataset, motorway_labels, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

def load_secondary_dataset(data_path='data'):
    # Function to load the secondary dataset only 

    with open(os.path.join(data_path,'secondary_dataset_window_64_proc_veh_DtA.pkl'), 'rb') as f:
        save = pickle.load(f, encoding='latin1')
        secondary_dataset = save['dataset']
        secondary_labels = save['labels']
        del save
        print('Secondary set', secondary_dataset.shape, secondary_labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(secondary_dataset, secondary_labels, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

def load_full_dataset(data_path='data'):
    # Function to load the full dataset (motorway+secondary roads)

    with open(os.path.join(data_path, 'motorway_dataset_window_64_proc_veh_DtA.pkl'), 'rb') as f:
        save = pickle.load(f, encoding='latin1')
        motorway_dataset = save['dataset']
        motorway_labels = save['labels']
        del save
        print('Motorway set', motorway_dataset.shape, motorway_labels.shape)

    with open(os.path.join(data_path,'secondary_dataset_window_64_proc_veh_DtA.pkl'), 'rb') as f:
        save = pickle.load(f, encoding='latin1')
        secondary_dataset = save['dataset']
        secondary_labels = save['labels']
        del save
        print('Secondary set', secondary_dataset.shape, secondary_labels.shape)

    dataset = np.concatenate((motorway_dataset,secondary_dataset), axis=0)
    labels = np.concatenate((motorway_labels,secondary_labels), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

class TimingCallback(Callback):
    def __init__(self):
        self.logs=[]
    def on_epoch_begin(self,epoch, logs={}):
        self.starttime=time()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(time()-self.starttime)

def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    filters = input.shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se

def load_dataset(dataset,config,hdc_encoded):
    """
    load the specific data set (from the data/ folder)
    @param dataset: specifies the data set [string]
    @param config: configure struct with necessary parameters [struct]
    @param hdc_encoded: if the data set to be loaded is to be HDC-coded [Bool]
    @return: set of training and test data [list]
    """
    # load preprocessed data
    if dataset == "full":
        if hdc_encoded==True:
            try:
                data = loadmat('data/preproc_data' + '_' + str(config.input_dim) + '_' + str(config.scale) + '.mat')
                X_train, X_test, y_train, y_test = train_test_split(
                    np.concatenate((data['output_motor'], data['output_second']),
                                   axis=0),
                    np.concatenate((data['motorway_labels'], data['secondary_labels']),
                                   axis=0), test_size=0.33, random_state=42)
            except:
                print('The dataset could not be loaded and may not have been preprocessed.')
        else:
            X_train, X_test, y_train, y_test = load_full_dataset()
    elif dataset == "motorway":
        if hdc_encoded==True:
            try:
                data = loadmat('data/preproc_data' + '_' + str(config.input_dim) + '_' + str(config.scale) + '.mat')
                X_train, X_test, y_train, y_test = train_test_split(data['output_motor'], data['motorway_labels'],
                                                                    test_size=0.33, random_state=42)
            except:
                print('The dataset could not be loaded and may not have been preprocessed.')
        else:
            X_train, X_test, y_train, y_test = load_motorway_dataset()
    elif dataset == "secondary":
        if hdc_encoded==True:
            try:
                data = loadmat('data/preproc_data' + '_' + str(config.input_dim) + '_' + str(config.scale) + '.mat')
                X_train, X_test, y_train, y_test = train_test_split(data['output_second'], data['secondary_labels'],
                                                                    test_size=0.33, random_state=42)
            except:
                print('The dataset could not be loaded and may not have been preprocessed.')
        else:
            X_train, X_test, y_train, y_test = load_secondary_dataset()
    elif dataset == "full_crossval":
        if hdc_encoded==True:
            try:
                data = loadmat('data/preproc_data' + '_' + str(config.input_dim) + '_' + str(config.scale) + '.mat')
            except:
                print('The dataset could not be loaded and may not have been preprocessed.')
            mot_data = data['output_motor']
            sec_data = data['output_second']
            mot_label = data['motorway_labels']
            sec_label = data['secondary_labels']
        else:
            data = loadmat('data/uah_dataset.mat')
            mot_data = data['motorway_dataset']
            sec_data = data['secondary_dataset']
            mot_label = data['motorway_labels']
            sec_label = data['secondary_labels']

        # creat cross-validation split (3-fold cross-validation) on the full dataset
        k = 3
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        #info = data['data_info']

        kf = KFold(n_splits=k)
        #motorway
        for train_idx, test_idx in kf.split(mot_data):
            X_train.append(mot_data[train_idx])
            y_train.append(mot_label[train_idx])
            X_test.append(mot_data[test_idx])
            y_test.append(mot_label[test_idx])
        k_idx = 0
        # secondary
        for train_idx, test_idx in kf.split(sec_data):
            X_train[k_idx]=np.concatenate((X_train[k_idx],sec_data[train_idx]),0)
            y_train[k_idx]=np.concatenate((y_train[k_idx],sec_label[train_idx]), 0)
            X_test[k_idx]=np.concatenate((X_test[k_idx],sec_data[test_idx]),0)
            y_test[k_idx]=np.concatenate((y_test[k_idx],sec_label[test_idx]), 0)
            k_idx+=1

    else:
        print("No valid dataset argument was set!")

    return X_train, X_test, y_train, y_test, config