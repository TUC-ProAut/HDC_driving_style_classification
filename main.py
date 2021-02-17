from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn import metrics
from sys import argv
from argparse import ArgumentParser
from config import Config
from utils import *
from model import HDC_ANN, LSTM_Network
from sklearn.metrics import classification_report
from scipy.io import savemat, loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from matplotlib import pyplot as plt
from datetime import datetime
import os
import logging
from time import time
import sklearn
import nengo_dl
import nengo


# config logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('main_log.log', 'a'))


def main_HDC(args):
    '''
    implementation of the HDC feed-forward network to predict the class of driving style
    - it uses preprocessed HDC encoding vectors
    '''
    # set config parameter
    config = Config()
    config.training_volume = args.training_volume
    config.input_dim = args.input_dim
    config.encoding_dim = args.encoding_dim
    config.scale = args.scale
    config.test_driver = args.test_driver
    tf.random.set_seed(0)

    # load data set
    data = load_dataset(args.dataset,config,True)
    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]
    config = data[4]

    # if train test data not a list, create one (full_crossval data set loading returns a list of splits - therefore we
    # are handling all training set as lists, even if they only contain one set)
    if type(X_train)==list:
        print("given data is not a list")
        X_train_list = X_train
        X_test_list = X_test
        y_train_list = y_train
        y_test_list = y_test
    else:
        X_train_list =[X_train]
        X_test_list = [X_test]
        y_train_list = [y_train]
        y_test_list = [y_test]

    # train for each element in list (that is why we need list form, even if it contains only one element)
    logger.info('Training data contains ' + str(len(X_train)) + ' training instances...')
    for it in range(len(X_train_list)):
        logger.info(('.......'))
        logger.info('instance ' + str(it) + ':')

        X_train = X_train_list[it]
        X_test = X_test_list[it]
        y_train = y_train_list[it]
        y_test = y_test_list[it]

        # use only fraction of training samples (if given)
        X_train = X_train[1:int(X_train.shape[0] * config.training_volume), :]
        y_train = y_train[1:int(y_train.shape[0] * config.training_volume)]

        # config.input_dim = X_train.shape[1]
        logger.info('Training dataset shape: ' + str(X_train.shape) + str(y_train.shape))
        logger.info('Test dataset shape: ' + str(X_test.shape) + str(y_test.shape))

        model = HDC_ANN(config)
        model.run_eagerly = True
        model.summary()

        cb_time = TimingCallback()
        history = model.fit(X_train, one_hot(y_train),
                            epochs=config.training_epochs,
                            batch_size=config.batch_size,
                            shuffle=True,
                            callbacks=[cb_time],
                            validation_data=(X_test, one_hot(y_test)))
        # log training time
        epoch_time = cb_time.logs
        mean_epoch_time = np.mean(epoch_time)
        overall_time = np.sum(epoch_time)
        logger.info("Mean Epoch time: " + str(mean_epoch_time))
        logger.info("overall training time: " + str(overall_time))

        # evaluate runtime and print results
        t = []
        for i in range(10):
            t1 = time()
            pred_test = model.predict(X_test)
            inference_time = time() - t1
            t.append((inference_time))

        logger.info("Inference time: " + str(np.mean(t)))
        pred_test_bool = np.argmax(pred_test, axis=1)

        logger.info('Accuracy on training data: ')
        report = classification_report(y_test, pred_test_bool, output_dict=True)
        logger.info(classification_report(y_test, pred_test_bool))

        predictions = pred_test.argmax(1)
        logger.info("Confusion matrix:")
        confusion_matrix = metrics.confusion_matrix(y_test, predictions)
        logger.info(confusion_matrix)

        model_summary_string = get_model_summary(model)
        # save as mat files
        save_dic = {"report": report, "confusion_matrix": confusion_matrix, "config": config, "pred": pred_test,
                    "label": y_test, 'model_summary': model_summary_string}
        savemat("results/" + args.dataset + "/results_HDC_" + str(config.input_dim) + "_" + str(config.scale) + "_" +
                str(config.encoding_dim) + '_' + str(config.training_volume) + ".mat", save_dic)

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("results/" + args.dataset + "/accuracy_history_HDC_" + str(config.input_dim) + "_" +
                    str(config.scale) + '_' + str(config.encoding_dim) + '_' + str(config.training_volume) + "instance_" + str(it) + ".png")
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("results/" + args.dataset + "/loss_history_HDC" + str(config.input_dim) + "_" +
                    str(config.scale) + '_' + str(config.encoding_dim) + '_' + str(config.training_volume) + "instance_" + str(it) + ".png")
        plt.show()

def main_Concat_ANN(args):
    '''
    implementation of a network that uses the concatenate sequences of al variables
    - input size of the network is 64*9 (64 timesteps and 9 sensors) = 576
    - network is the same as in HDC approach
    '''
    config = Config()
    config.training_volume = args.training_volume
    config.input_dim = args.input_dim
    config.encoding_dim = args.encoding_dim
    config.test_driver = args.test_driver
    tf.random.set_seed(0)

    # load preprocessed data
    data = load_dataset(args.dataset,config,False)
    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]
    config = data[4]

    # if train test data not a list, create one
    if type(X_train)==list:
        print("given data is not a list")
        X_train_list = X_train
        X_test_list = X_test
        y_train_list = y_train
        y_test_list = y_test
    else:
        X_train_list =[X_train]
        X_test_list = [X_test]
        y_train_list = [y_train]
        y_test_list = [y_test]

    # train for each element in list (that is why we need list form, even if it contains only one element)
    logger.info('Training data contains ' + str(len(X_train)) + ' training instances...')
    for it in range(len(X_train_list)):
        logger.info(('.......'))
        logger.info('instance ' + str(it) + ':')

        X_train = X_train_list[it]
        X_test = X_test_list[it]
        y_train = y_train_list[it]
        y_test = y_test_list[it]

        # use only fraction of training samples (if given)
        X_train = X_train[1:int(X_train.shape[0] * config.training_volume), :]
        y_train = y_train[1:int(y_train.shape[0] * config.training_volume), :]

        # concatenate the input data
        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        X_test = np.reshape(X_test, (X_test.shape[0], -1))

        config.input_dim = X_train.shape[1]

        logger.info('Training dataset shape: ' + str(X_train.shape) + str(y_train.shape))
        logger.info('Test dataset shape: ' + str(X_test.shape) + str(y_test.shape))

        model = HDC_ANN(config)
        model.summary()

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="auto")
        cb_time = TimingCallback()
        history = model.fit(X_train, one_hot(y_train),
                            epochs=config.training_epochs,
                            batch_size=config.batch_size,
                            shuffle=True,
                            callbacks=[cb_time],
                            validation_data=(X_test, one_hot(y_test)))

        # log training time
        epoch_time = cb_time.logs
        mean_epoch_time = np.mean(epoch_time)
        overall_time = np.sum(epoch_time)
        logger.info("Mean Epoch time: " + str(mean_epoch_time))
        logger.info("overall training time: " + str(overall_time))

        # evaluate and print results
        pred_test = model.predict(X_test)
        pred_test_bool = np.argmax(pred_test, axis=1)

        logger.info('Accuracy on training data: ')
        report = classification_report(y_test, pred_test_bool, output_dict=True)
        logger.info(classification_report(y_test, pred_test_bool))

        predictions = pred_test.argmax(1)
        logger.info("Confusion matrix:")
        confusion_matrix = metrics.confusion_matrix(y_test, predictions)
        logger.info(confusion_matrix)

        # save as mat files
        save_dic = {"report": report, "confusion_matrix": confusion_matrix, "config": config, "pred": pred_test,
                    "label": y_test}
        savemat("results/" + args.dataset + "/results_concatNet" + str(config.encoding_dim) + '_' +
                str(config.training_volume) + ".mat", save_dic)

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("results/" + args.dataset + "/accuracy_history_concatNet" + str(config.encoding_dim) + '_' +
                    str(config.training_volume) + "instance_" + str(it) + ".png")
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("results/" + args.dataset + "/loss_history_concatNet" + str(config.encoding_dim) + '_' +
                    str(config.training_volume) + "instance_" + str(it) + ".png")
        plt.show()

def main_orig(args):
    '''
    implementation of the original LSTM approach (https://github.com/KhaledSaleh/driving_behaviour_classification)
    '''
    # set config params specific for the original code
    config = Config()
    config.training_volume = args.training_volume
    config.input_dim = args.input_dim
    config.encoding_dim = args.encoding_dim
    config.scale = args.scale
    config.test_driver = args.test_driver
    tf.random.set_seed(0)

    # load preprocessed data
    data = load_dataset(args.dataset,config,False)
    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]
    config = data[4]

    logs = []
    # if train test data not a list, create one
    if type(X_train)==list:
        print("given data is not a list")
        X_train_list = X_train
        X_test_list = X_test
        y_train_list = y_train
        y_test_list = y_test
    else:
        X_train_list =[X_train]
        X_test_list = [X_test]
        y_train_list = [y_train]
        y_test_list = [y_test]

    # train for each element in list (that is why we need list form, even if it contains only one element)
    logger.info('Training data contains ' + str(len(X_train)) + ' training instances...')
    for it in range(len(X_train_list)):
        logger.info(('.......'))
        logger.info('instance ' + str(it) + ':')

        X_train = X_train_list[it]
        X_test = X_test_list[it]
        y_train = y_train_list[it]
        y_test = y_test_list[it]

        # use only fraction of training samples (if given)
        X_train = X_train[1:int(X_train.shape[0] * config.training_volume), :]
        y_train = y_train[1:int(y_train.shape[0] * config.training_volume), :]

        config.n_inputs = X_train.shape[2]
        config.train_count = len(X_train)
        config.test_data_count = len(X_test)
        config.n_steps = len(X_train[0])

        logger.info('Training dataset shape: ' + str(X_train.shape) + str(y_train.shape))
        logger.info('Test dataset shape: ' + str(X_test.shape) + str(y_test.shape))
        graph = tf.Graph()
        with graph.as_default():

            X = tf.compat.v1.placeholder(tf.float32, [None, config.n_steps, config.n_inputs], name="X")
            Y = tf.compat.v1.placeholder(tf.float32, [None, config.n_classes], name="Y")

            pred_Y = LSTM_Network(X, config)

            # Loss,optimizer,evaluation
            l2 = config.lambda_loss_amount * \
                 sum(tf.nn.l2_loss(tf_var) for tf_var in tf.compat.v1.trainable_variables())
            # Softmax loss and L2
            cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=pred_Y, labels=Y), name="cost") + l2
            optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=config.learning_rate).minimize(cost)

            correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

            saver = tf.compat.v1.train.Saver()

        with tf.compat.v1.Session(graph=graph, config=tf.compat.v1.ConfigProto(log_device_placement=False)) as sess:

            if not args.test:
                init_op = tf.compat.v1.global_variables_initializer()
                sess.run(init_op)
                best_accuracy = 0.0
                # Start training for each batch and loop epochs
                for i in range(config.training_epochs):
                    starttime = time()
                    for start, end in zip(range(0, config.train_count, config.batch_size),
                                          range(config.batch_size, config.train_count + 1, config.batch_size)):
                        sess.run(optimizer, feed_dict={X: X_train[start:end],
                                                       Y: one_hot(y_train[start:end],config.n_classes)})
                        saver.save(sess, os.path.join(args.save_dir, 'LSTM_model'))
                    # Test completely at every epoch: calculate accuracy
                    t1 = time()
                    pred_out, accuracy_out, loss_out = sess.run([pred_Y, accuracy, cost], feed_dict={
                        X: X_test, Y: one_hot(y_test,config.n_classes)})
                    inference_time = time() - t1
                    logs.append(time() - starttime)
                    print("Training iter: {},".format(i) + \
                          " Test accuracy : {},".format(accuracy_out) + \
                          " Loss : {}".format(loss_out))
                    best_accuracy = max(best_accuracy, accuracy_out)
                print("")
                mean_epoch_time = np.mean(logs)
                overall_time = np.sum(logs)
                logger.info("Mean Epoch time: " + str(mean_epoch_time))
                logger.info("overall training time: " + str(overall_time))
                logger.info("Final test accuracy: {}".format(accuracy_out))
                logger.info("Best epoch's test accuracy: {}".format(best_accuracy))

                print("")
            # start testing the trained model
            else:
                saver.restore(sess, os.path.join(args.save_dir, 'LSTM_model'))
                t1 = time()
                pred_out, accuracy_out, loss_out = sess.run([pred_Y, accuracy, cost], feed_dict={
                    X: X_test, Y: one_hot(y_test,config.n_classes)})
                inference_time = time() - t1
                print(" Test accuracy : {},".format(accuracy_out) + \
                      " Loss : {}".format(loss_out))

            predictions = pred_out.argmax(1)
            logger.info("Inference time: " + str(inference_time))
            report = classification_report(y_test, predictions, output_dict=True)
            confusion_matrix = metrics.confusion_matrix(y_test, predictions)
            logger.info(metrics.classification_report(y_test, predictions))
            logger.info(" Confusion Matrix: ")
            logger.info(metrics.confusion_matrix(y_test, predictions))

            # save as mat files
            save_dic = {"report": report, "confusion_matrix": confusion_matrix, "config": config, "pred": pred_out,
                        "label": y_test}
            savemat("results/" + args.dataset + "/results_origNet_" + str(config.training_volume) + ".mat", save_dic)

def main_SNN(args):
    config = Config()
    config.training_volume = args.training_volume
    config.input_dim = args.input_dim
    config.encoding_dim = args.encoding_dim
    config.scale = args.scale
    config.test_driver = args.test_driver

    # nego net params
    do_rate = 0.5
    num_epochs = 100
    enc_dim = 1000
    minibatch_size = 500
    seed = 0

    # load dataset
    data = load_dataset(args.dataset,config,True)
    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]
    config = data[4]

    # if train test data not a list, create one
    if type(X_train)==list:
        print("given data is not a list")
        X_train_list = X_train
        X_test_list = X_test
        y_train_list = y_train
        y_test_list = y_test
    else:
        X_train_list =[X_train]
        X_test_list = [X_test]
        y_train_list = [y_train]
        y_test_list = [y_test]

    # train for each element in list (that is why we need list form, even if it contains only one element)
    logger.info('Training data contains ' + str(len(X_train)) + ' training instances...')
    for it in range(len(X_train_list)):
        logger.info(('.......'))
        logger.info('instance ' + str(it) + ':')

        X_train = X_train_list[it]
        X_test = X_test_list[it]
        y_train = y_train_list[it]
        y_test = y_test_list[it]

        # use only fraction of training samples (if given)
        X_train = X_train[1:int(X_train.shape[0] * config.training_volume), :]
        y_train = y_train[1:int(y_train.shape[0] * config.training_volume)]

        y_train_oh = one_hot(y_train)
        y_test_oh = one_hot(y_test)

        # config.input_dim = X_train.shape[1]
        logger.info('Training dataset shape: ' + str(X_train.shape) + str(y_train.shape))
        logger.info('Test dataset shape: ' + str(X_test.shape) + str(y_test.shape))

        net = nengo.Network(seed=seed + 1)

        with net:
            # set some default parameters for the neurons that will make
            # the training progress more smoothly
            net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
            net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
            net.config[nengo.Connection].synapse = None
            neuron_type = nengo.LIF(amplitude=0.01)

            # this is an optimization to improve the training speed,
            # since we won't require stateful behaviour in this example
            nengo_dl.configure_settings(stateful=False)

            # the input node that will be used to feed in input vectors
            inp = nengo.Node(np.zeros(config.input_dim))

            x = nengo_dl.Layer(tf.keras.layers.Dropout(rate=do_rate))(inp)
            x = nengo_dl.Layer(neuron_type)(x)

            x = nengo_dl.Layer(tf.keras.layers.Dense(units=enc_dim))(x)
            x = nengo_dl.Layer(neuron_type)(x)

            out = nengo_dl.Layer(tf.keras.layers.Dense(units=len(y_train_oh[0])))(x)

            # we'll create two different output probes, one with a filter
            # (for when we're simulating the network over time and
            # accumulating spikes), and one without (for when we're
            # training the network using a rate-based approximation)
            out_p = nengo.Probe(out, label="out_p")
            out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")

        sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size, device="/gpu:0")

        # run training
        sim.compile(
            optimizer=tf.optimizers.RMSprop(0.001),
            loss={out_p: tf.losses.CategoricalCrossentropy(from_logits=True)},
        )

        # add single timestep to training data
        X_train = X_train[:, None, :]
        y_train_oh = y_train_oh[:, None]

        # when testing our network with spiking neurons we will need to run it
        # over time, so we repeat the input/target data for a number of
        # timesteps.
        n_steps = 30
        X_test = np.tile(X_test[:, None, :], (1, n_steps, 1))
        y_test_oh = np.tile(y_test_oh[:, None], (n_steps, 1))

        def classification_accuracy(y_true, y_pred):
            return tf.metrics.categorical_accuracy(y_true[:, -1], y_pred[:, -1])

        accuracy = sim.evaluate(X_test, {out_p_filt: y_test_oh}, verbose=0)["loss"],
        print("Accuracy before training:", accuracy)

        cb_time = TimingCallback()
        sim.fit(X_train,
                {out_p: y_train_oh},
                epochs=num_epochs,
                callbacks=[cb_time],
                )
        # log training time
        epoch_time = cb_time.logs
        mean_epoch_time = np.mean(epoch_time)
        training_time = np.sum(epoch_time)

        # save the parameters to file
        # sim.save_params("./nengo_dl_params")

        sim.compile(loss={out_p_filt: classification_accuracy})
        t1 = time()
        accuracy = sim.evaluate(X_test, {out_p_filt: y_test_oh}, verbose=0)["loss"],
        inference_time = time() - t1
        print("Accuracy after training:", accuracy)

        sim2 = nengo_dl.Simulator(net, minibatch_size=1, device="/gpu:0")

        y_pred = sim2.predict(X_test)
        y_pred_am = np.argmax(y_pred[out_p_filt][:,-1,:], axis=1)
        y_pred_am.shape = (y_pred_am.shape[0],1)
        f1_score = sklearn.metrics.f1_score(y_test,
                                            y_pred_am,
                                            average='micro')
        f1_score_weighted = sklearn.metrics.f1_score(
            y_test,
            y_pred_am,
            average='weighted'
        )

        logger.info("training time: " + str(training_time))
        logger.info("mean epoch time: " + str(mean_epoch_time))
        logger.info("Inference time: " + str(inference_time))


        print('Accuracy on training data: ')
        report = classification_report(y_test, y_pred_am, output_dict=True)
        logger.info(classification_report(y_test, y_pred_am))

        print("Confusion matrix:")
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred_am)
        print(confusion_matrix)

        # save as mat files
        save_dic = {"report": report, "confusion_matrix": confusion_matrix, "config": config, "pred": y_pred,
                    "label": y_test}
        savemat("results/" + args.dataset + "/results_Nengo_net_" + str(config.input_dim) + "_" + str(config.scale) + "_" +
                str(config.encoding_dim) + '_' + str(config.training_volume) + ".mat", save_dic)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', '-d',
                        help='Which split of the dataset to train/test the model on?' \
                             '(i.e. full, motorway, secondary or full_crossval)',
                        default='full')
    parser.add_argument('--save_dir', '-s',
                        help='Directory of (to be)-saved model',
                        default='saves')
    parser.add_argument('--hyperparams_experiment', '-f',
                        help='Run complete experiments (hyper-parameter analysis of HDC approach) if set to true',
                        type=bool,
                        default=False)
    parser.add_argument('--data_efficiency',
                        help='Run data efficiency experiments if set to true',
                        type=bool,
                        default=False)
    parser.add_argument('--test', action='store_true',
                        help='Start testing the saved model in $save_dir$ ' \
                             'othewrwise, it will start the training')
    parser.add_argument('--input_dim',
                        help='Defines the input dimension of the HDC model' \
                             '(possible values are 512, 1024 or 2048)',
                        default=2048)
    parser.add_argument('--scale',
                        help='scaling of the scalar encoding with fractional binding ' \
                             '(possible values are 2, 4, 6, 8 and 10)',
                        default=6)
    parser.add_argument('--encoding_dim',
                        help='dimension of the first hidden layer (named encoding dimension)' \
                             ' possible values are 20, 40, 60, 80, 100)',
                        default=40)
    parser.add_argument('--HDC_ANN',
                        help='Boolean value to train the HDC network (if true, the network ' \
                             'will be trained)',
                        type=bool,
                        default=False)
    parser.add_argument('--Concat_ANN',
                        help='Boolean value to train the concatenated network (if true, the network ' \
                             'will be trained)',
                        type=bool,
                        default=False)
    parser.add_argument('--LSTM',
                        help='Boolean value to train the LSTM network (if true, the network ' \
                             'will be trained)',
                        type=bool,
                        default=False)
    parser.add_argument('--HDC_SNN',
                        help='Boolean value to train the HDC network with SNN (if true, the network ' \
                             'will be trained)',
                        type=bool,
                        default=False)
    parser.add_argument('--training_volume',
                        help='To train with only a fraction of the training data. ' \
                             'Value is in range [0 1] (0 to 100 percentage).',
                        default=1)
    parser.add_argument('--test_driver',
                        default=1)
    args = parser.parse_args()

    # check if result folder exists
    if not os.path.exists('results/full'):
        os.makedirs('results/full')
    if not os.path.exists('results/motorway'):
        os.makedirs('results/motorway')
    if not os.path.exists('results/secondary'):
        os.makedirs('results/secondary')
    if not os.path.exists('results/full_crossval'):
        os.makedirs('results/full_crossval')

    # experiments to HDC classification
    n_dim = [512, 1024, 2048];
    scale = [2, 4, 6, 8, 10];
    encoding_dim = [20, 40, 60, 80, 100];

    training_volume = 1.0;
    training_volume_range = [0.2, 0.4, 0.6, 0.8, 1.0];

    logger.info('_________________________' + str(datetime.now()))

    # HDC network
    if args.HDC_ANN == True:
        logger.info("---HDC Model---")
        logger.info("- Dataset: " + args.dataset)
        # multiple experiments based on the HDC approach
        if args.hyperparams_experiment == True:
            logger.info("##### Full experiment (hyper-parameter analysis)")
            for d in range(len(n_dim)):
                for s in range(len(scale)):
                    for e in range(len(encoding_dim)):
                        args.input_dim = n_dim[d]
                        args.scale = scale[s]
                        args.encoding_dim = encoding_dim[e]
                        args.training_volume = training_volume

                        logger.info("Training with " + str(n_dim[d]) + " " + str(scale[s]) + " " + str(
                            encoding_dim[e]) + " training volume=" + str(training_volume))
                        main_HDC(args)
        elif args.data_efficiency == True:
            logger.info("#### Training efficiency:")
            logger.info("Config: input_dim = " + str(args.input_dim) + " scale = " + str(args.scale) +
                        " encoding_dim = " + str(args.encoding_dim))
            for t in range(len(training_volume_range)):
                args.training_volume = training_volume_range[t]
                logger.info("Training with training volume=" + str(training_volume_range[t]))
                main_HDC(args)
        else:
            logger.info("#### normal Training on " + args.dataset + ": ")
            logger.info("Config: input_dim = " + str(args.input_dim) + " scale = " + str(args.scale) +
                        " encoding_dim = " + str(args.encoding_dim) + " training_volume = " + str(args.training_volume))
            main_HDC(args)

    # concatenate intput network
    if args.Concat_ANN == True:
        logger.info("---Concat Model---")
        logger.info("- Dataset: " + args.dataset)
        logger.info("#### normal Training on" + args.dataset + ": ")
        main_Concat_ANN(args)

    # original LSTM network
    if args.LSTM == True:
        logger.info("---original LSTM Model---")
        logger.info("- Dataset: " + args.dataset)
        if args.data_efficiency == True:
            logger.info("#### Training efficiency:")
            for t in range(len(training_volume_range)):
                args.training_volume = training_volume_range[t]
                logger.info("Training with training volume=" + str(training_volume_range[t]))
                main_orig(args)
        else:
            logger.info("#### normal Training on " + args.dataset + ": ")
            main_orig(args)

    # SNN network
    if args.HDC_SNN == True:
        logger.info("---SNN Model---")
        logger.info("- Dataset: " + args.dataset)
        # multiple experiments based on the HDC approach
        if args.data_efficiency == True:
            logger.info("#### Training efficiency:")
            logger.info("Config: input_dim = " + str(args.input_dim) + " scale = " + str(args.scale) +
                        " encoding_dim = " + str(args.encoding_dim) + " training_volume = " + str(args.training_volume))
            for t in range(len(training_volume_range)):
                args.training_volume = training_volume_range[t]
                logger.info("Training with training volume=" + str(training_volume_range[t]))
                main_SNN(args)
        else:
            logger.info("#### normal Training on " + args.dataset + ": ")
            logger.info("Config: input_dim = " + str(args.input_dim) + " scale = " + str(args.scale) +
                        " encoding_dim = " + str(args.encoding_dim) + " training_volume = " + str(args.training_volume))
            main_SNN(args)
