# Dependencies
import numpy as np
import cupy as cp
import Util
from NeuralNet import *

class Config():
    """
    A config class specify the hyper-parameters.
    """

    def __init__(self):
        """
        This create a default configuration class.
        """
        self.layer_specs = [3072, 128, 10]
        self.activation = "tanh"
        self.learning_rate = 0.01
        self.batch_size = 128
        self.epochs = 100
        self.early_stop = True
        self.early_stop_threshold = 5
        self.L1_constant = 0
        self.L2_constant = 0
        self.momentum = False
        self.momentum_gamma = 0.9

    def __call__(self):
        """
        Directly display the configuration.
        """
        print("-"*15, "This display the configuration hyperparameter, change it direclty", "-"*15)
        print(f'self.layer_specs: {self.layer_specs}')
        print(f'self.activation: {self.activation}')
        print(f'self.learning_rate: {self.learning_rate}')
        print(f'self.batch_size: {self.batch_size}')
        print(f'self.epochs: {self.epochs}')
        print(f'self.early_stop: {self.early_stop}')
        print(f'self.early_stop_threshold: {self.early_stop_threshold}')
        print(f'self.L1_constant: {self.L1_constant}')
        print(f'self.L2_constant: {self.L2_constant}')
        print(f'self.momentum: {self.momentum}')
        print(f'self.momentum_gamma: {self.momentum_gamma}')

class Model():
    """
    Model Class includes a MLP Classifier.
    """

    def __init__(self, config):
        """
        Initilize the model
        """
        self.model = MLPClassifier(config)

    def train(x_train, y_train, x_valid, y_valid, config):
        """
        Train the model based on given training data and validation data.
        Learns the weight.
        Implement mini-batch SGD as learning method.
        Implement Early Stopping.

        args:
            model - an object of the NeuralNetwork class
            x_train - the train set examples 
            y_train - the test set targets/labels
            x_valid - the validation set examples
            y_valid - the validation set targets/labels

        returns:
            best_model - the best model we have
            train_loss_lst - the training loss recordings
            train_acc_lst - the training accuracy recordings
            val_loss_lst - the validation loss recordings
            val_acc_lst - the validation accuracy recordings
            best_stopping - the epochs that generate the best model
        """

        batch_size = config.batch_size
        early_stop = config.early_stop
        early_stop_epoch = config.early_stop_epoch

        # Specify the input