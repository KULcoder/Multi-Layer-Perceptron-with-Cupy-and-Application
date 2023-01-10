import copy
import os, gzip
import numpy as np
import cupy as cp
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# model helper functions
def load_config(path):
    """
    TODO: change this into a different style
    Loads the config yaml from the specified path

    args:
        path - Complete path of the config yaml file to be loaded
    returns:
        yaml - yaml object containing the config file
    """
    return None

def append_bias(X):
    """
    Appends bias to the input
    args:
        X (N X d 2D Array)
    returns:
        X_bias (N X (d+1)) 2D Array
    """
    
    return cp.append(X, cp.full((len(X), 1), 1), axis = 1) 


# Data preprocess Helper functions
def normalize_data(inp, axis=1):
    """
    Normalizes inputs (on per channel basis of every image) here to have 0 mean and unit variance.
    This will require reshaping to seprate the channels and then undoing it while returning

    args:
        inp : N X d 2D array where N is the number of examples and d is the number of dimensions
            d should be 1024 * 3 = 3072 for 3 channels for CIFAR-10 dataset
        axis: 0 or 1
            0: represents normalization among features / columnes
            1: represents normalization among instances / rows

    returns:
        normalized inp: N X d 2D array

    """
    channels = cp.hsplit(inp, 3) # split the data by 3 channels
    
    Z = []
    for channel in channels:
        u = cp.mean(channel, axis=axis) #calculate the mean
        sd = cp.std(channel, axis=axis) #calculate the SD
        Z_c = None
        if axis == 0:
            # the default dimension of numpy subtract boardcasting is along y
            Z_c = (channel - u)/sd #calculate the Z-score
        elif axis == 1:
            u = u.reshape(-1, 1)
            sd = sd.reshape(-1, 1)
            Z_c = (channel - u)/sd
        Z.append(Z_c)
    
    return np.concatenate(Z, axis=1) # concat the three channels back

def one_hot_encoding(array):
    """
    Efficient method doing one_hot_encoding on a 1-d numpy array. 
    
    Input:
        An 1-d numpy array needed to be one hot encoded.
    Output:
        An tuple contains:
            1. numpy matrix: the one-hot encoded result in a form as an  
                (do we need sparse matrix implementation?)
            2. python dictionary: mapping dictionary.
    """
    # Add an assertion on input type 
    assert type(array) == cp.ndarray, "Input must be in cupy array type"
    
    # get the unique values in the array
    unique_values, inverse_indices = cp.unique(array, return_inverse=True)
    
    # Initialize the one-hot encoded array and mapping dictionary
    one_hot_encoded_array = cp.zeros((len(array), len(unique_values)), dtype=cp.int8)
    mapping_dict = {}
    
    # Iterate over the unique values and build the one-hot encoded array and mapping dictionary
    for i, value in enumerate(unique_values):
        one_hot_encoded_array[:, i] = (inverse_indices == i)
        mapping_dict[value] = i
    
    # Return the one-hot encoded array and mapping dictionary
    return one_hot_encoded_array, mapping_dict



