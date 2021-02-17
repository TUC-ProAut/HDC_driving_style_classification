import tensorflow as tf  

class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self):

        # Trainging
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 2000
        self.batch_size = 1500
        self.training_volume = 1

        # network structure
        self.n_classes = 3
        self.n_steps = 64
        self.n_hidden = 100
        self.dropout = 0.75


