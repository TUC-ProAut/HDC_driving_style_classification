from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import Config
from utils import *
from model import HDC_ANN, HDC_ANN_tf
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.io import savemat, loadmat
from sklearn import metrics
from matplotlib import pyplot as plt
from datetime import datetime
from time import time
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import logging

# config logger
logger = logging.getLogger('log')

tf.compat.v1.disable_eager_execution()

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
    config.m = 0
    config.s = 1
    if args.runtime_measurement:
        config.n_time_measures = 10
    else:
        config.n_time_measures = 1
    # if dimension is smaller than 1000, set dropout to 0.5
    if args.input_dim<1000:
        config.dropout = 0.5

    # load data set
    data = load_dataset(args.dataset,config)
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
            y_train = y_train[1:int(y_train.shape[0] * config.training_volume)]

            logger.info('Training dataset shape: ' + str(X_train.shape) + str(y_train.shape))
            logger.info('Test dataset shape: ' + str(X_test.shape) + str(y_test.shape))

            config.n_classes = len(np.unique(y_train))
            config.n_inputs = X_train.shape[2]
            config.n_steps = X_train.shape[1]
            config.train_count = len(X_train)
            config.test_data_count = len(X_test)

            #######################################################################################
            # create HDC vectors (encoding)
            #######################################################################################
            tf.compat.v1.disable_eager_execution()
            # create HDC vectors
            t_train, X_train_HDC, traces_train, init_vecs = create_HDC_vectors(config, X_train)
            t_test, X_test_HDC, traces_test, init_vecs = create_HDC_vectors(config, X_test)
            preprocessing_time = t_train+t_test

            # normalize HDC encodings
            m = np.mean(X_train_HDC, axis=0)
            s = np.std(X_train_HDC,axis=0)
            config.m = m
            config.s = s
            X_train_HDC = np.divide(X_train_HDC - m,s)
            X_test_HDC = np.divide(X_test_HDC - m,s)

            #######################################################################################
            # keras model training
            #######################################################################################
            model = HDC_ANN(config)
            model.summary()

            # Create a TensorBoard callback
            logs = "logs/HDC_ts_" + datetime.now().strftime("%Y%m%d-%H%M%S")
            logs_pre = "logs/HDC_ts_preproc_" + datetime.now().strftime("%Y%m%d-%H%M%S")

            cb_time = TimingCallback()
            weight_fn = "./weights/HDC_ANN/%s_weights.h5" % args.dataset
            if not os.path.exists(weight_fn.rsplit('/', 1)[0]):
                os.makedirs(weight_fn.rsplit('/', 1)[0])
            model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode='auto',
                                               monitor='loss', save_best_only=True, save_weights_only=True)
            # compile model
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            if not args.test:
                # train model
                history = model.fit(X_train_HDC, to_categorical(y_train),
                                    epochs=config.training_epochs,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    callbacks=[cb_time, model_checkpoint],
                                    validation_data=(X_test_HDC, to_categorical(y_test)))

                # log training time
                epoch_time = cb_time.logs
                mean_epoch_time = np.mean(epoch_time)
                overall_time = np.sum(epoch_time)
                logger.info("Mean Epoch time: " + str(mean_epoch_time))
                logger.info("overall training time: " + str(overall_time))

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

            # load the best model weights
            model.load_weights(weight_fn)

            #############################################################################################
            # evaluation of results
            #############################################################################################
            # evaluate with tensorflow model (better comparability to LTSM TF model)
            X = tf.compat.v1.placeholder(tf.float32, [None, config.n_steps, config.n_inputs], name="X")
            # get weights of keras  model
            weights = model.get_weights()
            W = {'hidden': weights[0], 'output': weights[2]}
            biases = {'hidden': weights[1], 'output': weights[3]}
            # create TF model
            tf_model = HDC_ANN_tf(X, config, init_vecs, W, biases)
            t_i=[]
            for i in range(config.n_time_measures):
                sess = tf.compat.v1.Session()
                t1 = time()
                pred_test = sess.run(tf_model, feed_dict={X: X_test})
                inference_time = time() - t1
                t_i.append(inference_time)
            inference_time = np.median(t_i)

            logger.info("Preprocessing time for training: " + str(t_train))
            logger.info("Inference time: " + str(inference_time))
            logger.info("Inference time one sequence [ms]: " + str((inference_time*1000)/X_test.shape[0]))

            logger.info('Preprocessing time on dataset: ' + str(t_train) + ' + ' + str(t_test) + ' = ' + str(
                preprocessing_time))
            pred_test_bool = np.argmax(pred_test, axis=1)

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
    save_dic = {"report": report, "confusion_matrix": confusion_matrix, "config": config, "pred": pred_test,
                "label": y_test, "f1_mean": np.mean(f1_mean), "acc_mean": np.mean(acc_mean)}
    savemat("results/" + args.dataset + "/results_HDC_" + str(config.input_dim) + "_" + str(config.scale) + "_" +
            str(config.encoding_dim) + '_' + str(config.training_volume) + ".mat", save_dic)

    logger.info('Accuracy results of statistical repetitions: ' + str(acc_mean))
    logger.info('F1 scores of statistical repetitions: ' + str(f1_mean))

    # write all scores to extra file
    logger.info('Mean Score: ' + str(np.mean(f1_mean)))
    logger.info('Mean Accuracy: ' + str(np.mean(acc_mean)))
    with open("results/results_" + args.dataset + "_HDC.txt", 'a') as file:
        file.write(str(config.input_dim) + '\t'
                   + str(config.encoding_dim) + '\t'
                   + str(config.scale) + '\t'
                   + str(args.stat_iterations) + '\t'
                   + str(round(np.mean(f1_mean),3)) + '\t'
                   + str(round(np.mean(acc_mean),3)) + '\t'
                   + str(round(np.std(f1_mean),3)) + '\t'
                   + str(round(np.std(acc_mean),3)) + '\t'
                   + str(args.training_volume) + '\n'
                   )





