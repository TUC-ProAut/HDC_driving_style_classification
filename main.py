from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser
from datetime import datetime
import logging

from utils import *
from HDC_ANN import main_HDC
from HDC_SNN import main_SNN
from LSTM import main_LSTM
from Concat_ANN import main_Concat_ANN


# config logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('log')
if not os.path.exists("./logs"):
    os.makedirs("./logs")
logger.addHandler(logging.FileHandler('./logs/main_log.log', 'a'))
stdout_handler = logging.StreamHandler()
logger.addHandler(stdout_handler)


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
    parser.add_argument('--runtime_measurement',
                        help='Run inference time measurement with 10 repetitions.',
                        type=bool,
                        default=False)
    parser.add_argument('--test',
                        help='Start testing the saved model in $save_dir$ ' \
                             'othewrwise, it will start the training',
                        type=bool,
                        default=False)
    parser.add_argument('--input_dim',
                        help='Defines the input dimension of the HDC model' \
                             '(possible values are 512, 1024 or 2048)',
                        type=int,
                        default=2048)
    parser.add_argument('--scale',
                        help='scaling of the scalar encoding with fractional binding ' \
                             '(possible values are 2, 4, 6, 8 and 10)',
                        type=int,
                        default=6)
    parser.add_argument('--encoding_dim',
                        help='dimension of the first hidden layer (named encoding dimension)' \
                             ' possible values are 20, 40, 60, 80, 100)',
                        type=int,
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
                        type=float,
                        default=1)
    parser.add_argument('--stat_iterations',
                        help="Number of repetitions",
                        default=1,
                        type=int)
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
    n_dim = [512, 1024, 2048]
    scale = [2, 4, 6, 8, 10]
    encoding_dim = [20, 40, 60, 80, 100]

    training_volume = 1.0
    training_volume_range = [0.2, 0.4, 0.6, 0.8, 1.0]

    logger.info('_________________________' + str(datetime.now()))

    # HDC network
    if args.HDC_ANN:
        logger.info("---HDC Model---")
        logger.info("- Dataset: " + args.dataset)
        # multiple experiments based on the HDC approach
        if args.hyperparams_experiment:
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
        elif args.data_efficiency:
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
    if args.Concat_ANN:
        logger.info("---Concat Model---")
        logger.info("- Dataset: " + args.dataset)
        logger.info("#### normal Training on" + args.dataset + ": ")
        main_Concat_ANN(args)

    # original LSTM network
    if args.LSTM:
        logger.info("---original LSTM Model---")
        logger.info("- Dataset: " + args.dataset)
        if args.data_efficiency:
            logger.info("#### Training efficiency:")
            for t in range(len(training_volume_range)):
                args.training_volume = training_volume_range[t]
                logger.info("Training with training volume=" + str(training_volume_range[t]))
                main_LSTM(args)
        else:
            logger.info("#### normal Training on " + args.dataset + ": ")
            main_LSTM(args)

    # SNN network
    if args.HDC_SNN:
        logger.info("---SNN Model---")
        logger.info("- Dataset: " + args.dataset)
        # multiple experiments based on the HDC approach
        if args.data_efficiency:
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
