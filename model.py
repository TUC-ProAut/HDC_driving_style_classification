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
    encoding_dim = config.encoding_dim

    # This is our input image
    input = keras.Input(shape=(config.input_dim,))
    dropout = layers.Dropout(config.dropout)(input)
    encoded = layers.Dense(encoding_dim, activation='relu')(dropout)
    output = layers.Dense(config.n_classes, activation='softmax')(encoded)

    # This model maps an input to its reconstruction
    model = keras.Model(input, output)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


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

