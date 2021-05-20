from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import Config
from utils import *
from model import HDC_ANN, LSTM_Network
from sklearn.metrics import classification_report, f1_score
from scipy.io import savemat, loadmat
from sklearn import metrics
import os
from time import time
from tensorflow.python.client import timeline
import logging
import matplotlib.pyplot as plt

# config logger
logger = logging.getLogger('log')


def main_LSTM(args):
    '''
    implementation of the original LSTM approach (https://github.com/KhaledSaleh/driving_behaviour_classification)
    '''
    # set config params specific for the original code
    config = Config()
    config.training_volume = args.training_volume
    config.input_dim = args.input_dim
    config.encoding_dim = args.encoding_dim
    config.scale = args.scale
    if args.runtime_measurement:
        config.n_time_measures = 10
    else:
        config.n_time_measures = 1

    # load preprocessed data
    data = load_dataset(args.dataset,config)
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

    #######################################################################################
    # statistical iteration
    #######################################################################################
    acc_mean = []
    f1_mean = []

    for stat_it in range(args.stat_iterations):
        logger.info('Statistial iteration: ' + str(stat_it))

        # train for each element in list (that is why we need list form, even if it contains only one element)
        logger.info('Training data contains ' + str(len(X_train_list)) + ' training instances...')
        scores = []
        accs = []
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
            config.n_classes = len(np.unique(y_train))

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
                            saver.save(sess, os.path.join("./weights", 'LSTM_model'))
                        # Test completely at every epoch: calculate accuracy
                        pred_out, accuracy_out, loss_out = sess.run([pred_Y, accuracy, cost], feed_dict={
                            X: X_test, Y: one_hot(y_test, config.n_classes)})
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
                    saver.restore(sess, os.path.join("./weights", 'LSTM_model'))
                    t1 = time()
                    pred_out, accuracy_out, loss_out = sess.run([pred_Y, accuracy, cost], feed_dict={
                        X: X_test, Y: one_hot(y_test,config.n_classes)})
                    inference_time = time() - t1
                    print(" Test accuracy : {},".format(accuracy_out) + \
                          " Loss : {}".format(loss_out))

            #############################################################################################
            # evaluation of results
            #############################################################################################

            pred_test_bool = pred_out.argmax(1)

            # runtime measurement
            t=[]
            traces = []
            options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            run_metadata = tf.compat.v1.RunMetadata()
            for i in range(config.n_time_measures):
                with tf.compat.v1.Session(graph=graph, config=tf.compat.v1.ConfigProto(log_device_placement=False)) as Sess:
                    init_op = tf.compat.v1.global_variables_initializer()
                    Sess.run(init_op)
                    t1 = time()
                    Sess.run([pred_Y, accuracy, cost], feed_dict={
                        X: X_test, Y: one_hot(y_test, config.n_classes)}, options=options, run_metadata=run_metadata)
                    inference_time = time() - t1
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    traces.append(chrome_trace)
                    t.append(inference_time)
            with open('./logs/LSTM_ts_preproc_timeline_test.json', 'w') as f:
                f.write(traces[-1])
            inference_time = np.median(inference_time)
            logger.info("Inference time: " + str(inference_time))
            logger.info("Inference time of one sequence [ms]: " + str(inference_time*1000/X_test.shape[0]))

            logger.info('Accuracy on training data: ')
            report = classification_report(y_test.astype(int), pred_test_bool, output_dict=True)
            logger.info(classification_report(y_test.astype(int), pred_test_bool))

            accs.append((report['accuracy']))

            logger.info("Confusion matrix:")
            confusion_matrix = metrics.confusion_matrix(y_test.astype(int), pred_test_bool)
            logger.info(confusion_matrix)

            # f1 score
            f1 = f1_score(y_test.astype(int), pred_test_bool, average='weighted')
            scores.append(f1)
            logger.info("F1 Score: " + str(f1))

        # add results to statistical result array
        acc_mean.append(np.mean(accs))
        f1_mean.append(np.mean(scores))

    # save as mat files
    save_dic = {"report": report, "confusion_matrix": confusion_matrix, "config": config, "pred": pred_out,
                "label": y_test, "f1": np.mean(f1_mean), "acc_mean": np.mean(acc_mean)}
    savemat("results/" + args.dataset + "/results_origNet_" + str(config.training_volume) + ".mat", save_dic)

    logger.info('Accuracy results of statistical repetitions: ' + str(acc_mean))
    logger.info('F1 scores of statistical repetitions: ' + str(f1_mean))

    # write all scores to extra file
    logger.info('Mean Score: ' + str(np.mean(f1_mean)))
    logger.info('Mean Accuracy: ' + str(np.mean(acc_mean)))
    with open("results/results_" + args.dataset + "_LSTM.txt", 'a') as file:
        file.write(str(args.stat_iterations) + '\t'
                   + str(round(np.mean(f1_mean), 3)) + '\t'
                   + str(round(np.mean(acc_mean), 3)) + '\t'
                   + str(round(np.std(f1_mean), 3)) + '\t'
                   + str(round(np.std(acc_mean), 3)) + '\t'
                   + str(args.training_volume) + '\n'
                   )



