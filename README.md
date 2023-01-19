# Multi-Layer-Perceptron-with-Numpy-and-Application

*This project is inspired from a deep learning course in University of California, San Diego. All rights reserved*.

This is an implementation of <u>Multi-Layer-Perceptron</u> model with <u>Stochastic Mini-batch Gradient Descent</u> with `CuPy`. And designed to be easy to reuse.

## Required Dependencies (Packages)

For acceleration concern, this project uses `CUPY`, which is a GPU accelerated `NumPy` for the training process. In theory, all the `CuPy` method in this directly replaced by their corresponding `NumPy` method.

- `numpy`
- `cupy`
- some other python basic libraries

## How to Run

1. Copy those files to the root directory

   - NeuralNet.py
   - Util.py
   - PersonalMLP.py

2. Following such template:

   ```python
   import PersonalMLP
   
   # load data 
   X_train = ...
   y_train = ...
   X_test = ...
   y_test = ...
   X_valid = ...
   y_valid = ...
   
   # ensure to normalize the X data, as a requirement of neural network
   
   # configs
   config = personalMLP.Config()
   # check configs
   config()
   # change configs
   config.hyperparameter1 = specific_value
   config.hyperparameter2 = specific_value
   
   # set up the model
   model = PersonalMLP.Model(config)
   
   # train the model
   statistics = model.train(X_train, y_train, X_valid, y_valid)
   
   # use the model to generate predict test
   y_test_pred = model.forward(X_test)
   
   # calculate matrics and demonstrate result
   ```

**An example of how to use this code is in the Example Jupyter Notebook file**.

## Implementation

### Back-Propagation

The foundation to all modern deep neural network: back propagation. Within this project, the back-propagation is plainly implemented by `CuPy` following below equations:

- Learning Rule:
  
  $$
  w_{ij} = w_{ij} - \alpha \frac{\part J}{\part w_{ij}} = w_{ij} +\alpha \delta_j z_i
  $$

- Definition on $\delta$:

  - When $j$ is for the output layer units
    
    $$
    \delta_j = t_j - y_j
    $$

  - When $j$ is for the hidden layer units
    
    $$
    \delta_j = g'(a_j)\sum_k \delta_k w_{jk}
    $$
    

