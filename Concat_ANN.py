from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import Config
from utils import *
from model import HDC_ANN
from sklearn.metrics import classification_report, f1_score
from scipy.io import savemat, loadmat
from sklearn import metrics
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import logging

# config logger
logger = logging.getLogger('log')

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
        logger.info('Training data contains ' + str(len(X_train)) + ' training instances...')
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

            # concatenate the input data
            X_train = np.reshape(X_train, (X_train.shape[0], -1))
            X_test = np.reshape(X_test, (X_test.shape[0], -1))

            config.input_dim = X_train.shape[1]

            logger.info('Training dataset shape: ' + str(X_train.shape) + str(y_train.shape))
            logger.info('Test dataset shape: ' + str(X_test.shape) + str(y_test.shape))

            config.n_classes = len(np.unique(y_train))

            #######################################################################################
            # keras model training
            #######################################################################################

            model = HDC_ANN(config)
            model.summary()

            cb_time = TimingCallback()
            weight_fn = "./weights/Concat_ANN/%s_weights.h5" % args.dataset
            if not os.path.exists(weight_fn.rsplit('/', 1)[0]):
                os.makedirs(weight_fn.rsplit('/', 1)[0])
            model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode='auto',
                                               monitor='loss', save_best_only=True, save_weights_only=True)

            # compile model
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            history = model.fit(X_train, to_categorical(y_train),
                                epochs=config.training_epochs,
                                batch_size=config.batch_size,
                                shuffle=True,
                                callbacks=[cb_time, model_checkpoint],
                                validation_data=(X_test, to_categorical(y_test)))

            # log training time
            epoch_time = cb_time.logs
            mean_epoch_time = np.mean(epoch_time)
            overall_time = np.sum(epoch_time)
            logger.info("Mean Epoch time: " + str(mean_epoch_time))
            logger.info("overall training time: " + str(overall_time))

            # load the best model weights
            model.load_weights(weight_fn)

            #############################################################################################
            # evaluation of results
            #############################################################################################

            # evaluate and print results
            pred_test = model.predict(X_test)
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

            # save as mat files
            save_dic = {"report": report, "confusion_matrix": confusion_matrix, "config": config, "pred": pred_test,
                        "label": y_test, "f1": f1}
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

        # add results to statistical result array
        acc_mean.append(np.mean(accs))
        f1_mean.append(np.mean(scores))

    logger.info('Accuracy results of statistical repetitions: ' + str(acc_mean))
    logger.info('F1 scores of statistical repetitions: ' + str(f1_mean))

    # write all scores to extra file
    logger.info('Mean Score: ' + str(np.mean(f1_mean)))
    logger.info('Mean Accuracy: ' + str(np.mean(acc_mean)))
    with open("results/results_" + args.dataset + "_Concat.txt", 'a') as file:
        file.write(str(config.input_dim) + '\t'
                   + str(config.encoding_dim) + '\t'
                   + str(args.stat_iterations) + '\t'
                   + str(round(np.mean(f1_mean), 3)) + '\t'
                   + str(round(np.mean(acc_mean), 3)) + '\t'
                   + str(round(np.std(f1_mean), 3)) + '\t'
                   + str(round(np.std(acc_mean), 3)) + '\n'
                   )