import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.layers import *
import keras.backend as K
from utils import *


def HDC_ANN(config):
    """HDC feed-forward model
    """
    tf.config.optimizer.set_jit(True)
    encoding_dim = config.encoding_dim

    input = keras.Input(shape=(config.input_dim,))
    dropout = layers.Dropout(config.dropout)(input)
    encoded = layers.Dense(encoding_dim, activation='relu')(dropout)
    output = layers.Dense(config.n_classes, activation='softmax')(encoded)

    model = keras.Model(input, output)

    return model

def HDC_ANN_tf(input, config, init_vecs, W, biases):
    """ Tensorflow model of the HDC ANN """
    tf.config.optimizer.set_jit(True)

    # preprocessing
    preproc = HDC_tf_preproc(input, init_vecs)

    # normalize data
    norm_data = tf.divide(preproc - config.m,config.s)

    # neural network for classification
    #hidden = tf.nn.dropout(preproc, rate=config.dropout)
    hidden = tf.matmul(norm_data,W['hidden']) + biases['hidden']
    hidden = tf.nn.relu(hidden)

    output = tf.matmul(hidden,W['output']) + biases['output']
    output = tf.nn.softmax(output)

    print("total # of trainable parameters:" + str(
        np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])))

    return output

def LSTM_Network(feature_mat, config):
    """model a LSTM Network, (borrowed from https://github.com/KhaledSaleh/driving_behaviour_classification)
      it stacks 2 LSTM layers, each layer has n_hidden=32 cells
       and 1 output layer, it is a full connet layer
      argument:
        feature_mat: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : matrix  output shape [batch_size,n_classes]
    """
    W = {
        'hidden': tf.Variable(tf.random.normal([config.n_inputs, config.n_hidden]), name="W_hidden"),
        'output': tf.Variable(tf.random.normal([config.n_hidden, config.n_classes]), name="W_output")
    }
    biases = {
        'hidden': tf.Variable(tf.random.normal([config.n_hidden], mean=1.0), name="b_hidden"),
        'output': tf.Variable(tf.random.normal([config.n_classes]), name="b_output")
    }

    feature_mat = tf.transpose(feature_mat, [1, 0, 2])
    feature_mat = tf.reshape(feature_mat, [-1, config.n_inputs], name="features_reshape")
    hidden = tf.nn.relu(tf.matmul(
        feature_mat, W['hidden']
    ) + biases['hidden'])
    hidden = tf.split(hidden, config.n_steps, 0, name="input_hidden")
    lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=1.0)
    lsmt_layers = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)
    outputs, _ = tf.compat.v1.nn.static_rnn(lsmt_layers, hidden, dtype=tf.float32)
    lstm_last_output = outputs[-1]
    # Linear activation
    final_out = tf.add(tf.matmul(lstm_last_output, W['output']), biases['output'], name="logits")

    print("total # of trainable parameters:" + str(
        np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])))

    return final_out

def HDC_tf_preproc(inputs, init_vecs):
    '''
    preprocessing function to create HDC vectors with tensorflow on GPU
    @param inputs: input tensor (#samples , #variables, #timesteps)
    @param init_vecs: initial hypervectors
    @return: context vectors
    '''
    init_vec = init_vecs['init_vec']
    sensor_ids = init_vecs['sensor_ids']
    timestamps = init_vecs['timestamps']
    scale = init_vecs['scale']

    tf.config.optimizer.set_jit(True)
    # fractional binding
    reshaped_input = tf.tile(tf.expand_dims(inputs, axis=3), [1,1,1,init_vec.shape[0]])

    expand_init_vec = tf.expand_dims(K.expand_dims(tf.transpose(init_vec * scale), axis=0), axis=0)
    reshaped_init_vec = tf.tile(expand_init_vec, [1,reshaped_input.shape[1],1,1])
    reshaped_init_vec = tf.tile(reshaped_init_vec, [1,1,reshaped_input.shape[2],1])
    # fractional binding with scale
    encoded_scalars = tf.math.multiply(reshaped_input, reshaped_init_vec)

    # bind to sensors
    sensor_ids = tf.transpose(sensor_ids, (1, 0))
    sensor_ids = tf.expand_dims(tf.expand_dims(sensor_ids, axis=0), axis=0)
    sensor_ids = tf.tile(sensor_ids, [1,encoded_scalars.shape[1],1,1])
    sensor_vals = tf.add(encoded_scalars, sensor_ids)

    # bundle all sensor vectors
    vals_cos = tf.cos(sensor_vals)
    vals_sin = tf.sin(sensor_vals)
    sensor_bundle_cos = (tf.reduce_sum(vals_cos, axis=2))
    sensor_bundle_sin = (tf.reduce_sum(vals_sin, axis=2))
    sensor_bundle = tf.math.atan2(sensor_bundle_sin, sensor_bundle_cos)

    # encode temporal context
    timestamps = tf.transpose(timestamps, (1, 0))
    timestamps = tf.expand_dims(timestamps, axis=0)
    context_vecs = tf.add(sensor_bundle, timestamps)

    # bundle temporal context
    complex_context_cos = tf.cos(context_vecs)
    complex_context_sin = tf.sin(context_vecs)
    context_bundle_cos = (tf.reduce_sum(complex_context_cos, axis=1))
    context_bundle_sin = (tf.reduce_sum(complex_context_sin, axis=1))
    context_bundle = tf.math.atan2(context_bundle_sin, context_bundle_cos)

    return context_bundle