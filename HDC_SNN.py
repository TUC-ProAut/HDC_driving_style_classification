from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import metrics
from config import Config
from utils import *
from sklearn.metrics import classification_report
from scipy.io import savemat
from sklearn import metrics
import logging
from time import time
import sklearn
import nengo_dl
import nengo

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

# config logger
logger = logging.getLogger('log')

tf.compat.v1.enable_eager_execution()

def main_SNN(args):
    config = Config()
    config.training_volume = args.training_volume
    config.input_dim = args.input_dim
    config.encoding_dim = args.encoding_dim
    config.scale = args.scale
    if args.runtime_measurement:
        config.n_time_measures = 10
    else:
        config.n_time_measures = 1

    # nego net params
    do_rate = 0.5
    num_epochs = 200
    enc_dim = 1000
    minibatch_size = 500
    seed = 0

    # load dataset
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
        seed = stat_it

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
            X_train = X_train[1:int(X_train.shape[0] * config.training_volume), :,:]
            y_train = y_train[1:int(y_train.shape[0] * config.training_volume)]

            y_train_oh = one_hot(y_train)
            y_test_oh = one_hot(y_test)

            # config.input_dim = X_train.shape[1]
            logger.info('Training dataset shape: ' + str(X_train.shape) + str(y_train.shape))
            logger.info('Test dataset shape: ' + str(X_test.shape) + str(y_test.shape))

            config.n_classes = len(np.unique(y_train))
            config.n_inputs = X_train.shape[2]
            config.n_steps = X_train.shape[1]

            #######################################################################################
            # create HDC vectors (encoding)
            #######################################################################################
            # create HDC vectors
            t_train, X_train, traces_train, init_vecs = create_HDC_vectors(config, X_train)
            t_test, X_test, traces_test, init_vecs = create_HDC_vectors(config, X_test)
            preprocessing_time = t_train + t_test

            # normalize HDC encodings
            m = np.mean(X_train, axis=0)
            s = np.std(X_train,axis=0)
            config.m = m
            config.s = s
            X_train = np.divide(X_train - m,s)
            X_test = np.divide(X_test - m,s)

            #######################################################################################
            # nengo model training
            #######################################################################################

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

            #############################################################################################
            # evaluation of results
            #############################################################################################

            sim.compile(loss={out_p_filt: classification_accuracy})

            # runtime measurement
            t=[]
            for i in range(config.n_time_measures):
                t1 = time()
                accuracy = sim.evaluate(X_test, {out_p_filt: y_test_oh}, verbose=0)["loss"],
                inference_time = time() - t1
                t.append(inference_time)
            inference_time = np.mean(t)
            print("Accuracy after training:", accuracy)
            accs.append(accuracy)

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

            logger.info("Training time: " + str(training_time))
            logger.info("Mean epoch time: " + str(mean_epoch_time))
            logger.info("Inference time: " + str(inference_time))

            logger.info("Preprocessing time for training: " + str(t_train))
            logger.info("Preprocessing time for testing: " + str(t_test))
            logger.info("Inference time one sequence [ms]: " + str((inference_time*1000+t_test*1000)/X_test.shape[0]))


            print('Accuracy on training data: ')
            report = classification_report(y_test, y_pred_am, output_dict=True)
            logger.info(classification_report(y_test, y_pred_am))

            print("Confusion matrix:")
            confusion_matrix = metrics.confusion_matrix(y_test, y_pred_am)
            print(confusion_matrix)

            # f1 score
            scores.append(f1_score_weighted)
            logger.info("F1 Score: " + str(f1_score_weighted))

            # close simulator
            sim.close()
            sim2.close()


        # add results to statistical result array
        acc_mean.append(np.mean(accs))
        f1_mean.append(np.mean(scores))

    # save as mat files
    save_dic = {"report": report, "confusion_matrix": confusion_matrix, "config": config, "pred": y_pred,
                "label": y_test, "f1_mean": np.mean(f1_mean)}
    savemat("results/" + args.dataset + "/results_Nengo_net_" + str(config.input_dim) + "_" + str(config.scale) + "_" +
            str(config.encoding_dim) + '_' + str(config.training_volume) + ".mat", save_dic)

    logger.info('Accuracy results of statistical repetitions: ' + str(acc_mean))
    logger.info('F1 scores of statistical repetitions: ' + str(f1_mean))

    # write all scores to extra file
    logger.info('Mean Score: ' + str(np.mean(f1_mean)))
    logger.info('Mean Accuracy: ' + str(np.mean(acc_mean)))
    with open("results/results_" + args.dataset + "_SNN.txt", 'a') as file:
        file.write(str(config.input_dim) + '\t'
                   + str(config.encoding_dim) + '\t'
                   + str(config.scale) + '\t'
                   + str(args.stat_iterations) + '\t'
                   + str(round(np.mean(f1_mean),3)) + '\t'
                   + str(round(np.mean(acc_mean),3)) + '\t'
                   + str(round(np.std(f1_mean),3)) + '\t'
                   + str(round(np.std(acc_mean),3)) + '\n'
                   )