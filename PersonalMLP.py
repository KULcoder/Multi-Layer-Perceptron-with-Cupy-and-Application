# Dependencies
import numpy as np
import cupy as cp
import Util
import copy
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

    def train(self, X_train, y_train, X_valid, y_valid, config):
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
        epochs = config.epochs
        early_stop = config.early_stop
        early_stop_epoch = config.early_stop_epoch

        # prepare the decode label for checking accuracy
        y_valid_decode = Util.onehot_decode(y_valid)

        # change the validation set into cupy array for validation usage
        X_valid = cp.array(X_valid)
        y_valid = cp.array(y_valid)

        # statistics
        val_loss_lst = []
        train_loss_lst = []
        val_acc_lst = []
        train_acc_lst = []

        # early stopping use
        min_val_loss = np.inf
        best_model = None
        best_stopping = 0
        worse_epochs_num = 0

        for epoch in range(epochs):
            # one epoch of the training

            # shuffle the data
            shuffler = np.random.permutation(len(X_train))
            X_train, y_train = X_train[shuffler], y_train[shuffler]

            # minibatches iterator
            minibatches = Util.generate_minibatches([X_train, y_train], batch_size)

            # minibatches training processes
            epoch_train_loss = []
            epoch_train_acc = []
            for X_batch, y_batch in minibatches:
                # one minibatch

                # to Cupy
                X_batch = cp.array(X_batch)
                y_batch = cp.array(y_batch)

                # train: forward and backward
                train_loss = self.model.forward(X_batch, y_batch)
                self.model.backward(y_batch)
                epoch_train_loss.append(cp.asnumpy(train_loss)) # record the train loss

                # accuracy
                y_batch_pred_decode = Util.one_hot_decode(cp.asnumpy(self.model.y))
                y_batch_decode = Util.onehot_decode(cp.asnumpy(y_batch))
                train_acc = Util.calculateAcc(y_batch_pred_decode, y_batch_decode)
                epoch_train_acc.append(train_acc) # record the train accuracy

            # at the end of one epoch:

            # append the average epochs train lost and accuracy
            train_loss_lst.append(np.average(epoch_train_loss))
            train_acc_lst.append(np.average(epoch_train_acc))

            # run the forward with validation set and calculate statistics
            val_loss = self.model.forward(X_valid, y_valid)
            val_loss_lst.append(cp.asnumpy(val_loss))
            # get the accuracy
            y_pred_decode = Util.onehot_decode(cp.asnumpy(self.model.y)) # directly decode from the softmax
            accuracy = Util.calculateAcc(y_pred_decode, y_valid_decode)
            val_acc_lst.append(accuracy)

            if early_stop:
                # if early_stop is required

                if val_loss < min_val_loss:
                    min_val_loss = val_loss # record the minimial loss value
                    best_model = copy.deepcopy(self.model) # record the best model
                    best_stopping = epoch # record the epoch that generate the best loss value
                    worse_epochs_num = 0 # if best is regenerated, initlize the flag
                else:
                    # if the performance decrease
                    worse_epochs_num += 1

                if worse_epochs_num >= early_stop_epoch:
                    # when consecutively perform worse in some number
                    break

        # release the cupy memory
        del(X_valid)
        del(y_valid)

        if early_stop:
            return best_model, train_loss_lst, train_acc_lst, val_loss_lst, val_acc_lst, best_stopping
        else:
            return self.model, train_loss_lst, train_acc_lst, val_loss_lst, val_acc_lst, best_stopping
