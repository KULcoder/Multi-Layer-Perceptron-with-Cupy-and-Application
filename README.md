# Multi-Layer-Perceptron-with-Cupy-and-Application

*This project is inspired from a deep learning course in University of California, San Diego. All rights reserved*. 

This is an implementation of <u>Multi-Layer-Perceptron</u> model with <u>Stochastic Mini-batch Gradient Descent</u> with `CuPy`. And designed to be easy to reuse.

## Required Dependencies (Packages)

For acceleration concern, this project uses `CuPy`, which is a GPU accelerated `NumPy` for the training process. In theory, all the `CuPy` method in this project can be directly replaced by their corresponding `NumPy` method.

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

## List of Default Hyperparameter

```python
layer_specs: [3072, 128, 10]
activation: tanh
learning_rate: 0.01
batch_size: 128
epochs: 100
early_stop: True
early_stop_threshold: 5
L1_constant: 0
L2_constant: 0
momentum: False
momentum_gamma: 0.9
```

## Implementation

### Back-Propagation

The foundation to all modern deep neural network: back propagation. Within this project, the back-propagation is plainly implemented by `CuPy` following below equations:

Learning Rule:

$$
w_{ij} = w_{ij} - \alpha \frac{\partial J}{\partial w_{ij}} = w_{ij} +\alpha \delta_j z_i
$$

Definition on $\delta$ (recursively):

- When $j$ is for the output layer units

$$
 \delta_j = t_j - y_j
$$

- When $j$ is for the hidden layer units

$$
 \delta_j = g'(a_j)\sum_k \delta_k w_{jk}
$$

### Activation Function

This project enables three types of activation functions: 'sigmoid', 'tanh', 'ReLU'.

### Stochastic Gradient Descent

There are many learning rule (optimizer) methods for deep learning. This project simply utilize the Mini-Batch Stochastic Gradient Descent.

#### Early Stopping Mechanism

This project allows early stopping when seeing worsen performance more than a setting times and return the best model (which is the model with least validation loss).

#### Cross Entropy Loss

As discussed by Maximum Likelihood Estimation, the correct loss function chosen for multi-class classification task is cross-entropy loss.

### Momentum Method

Allowing gradient to be affect by accumulated gradient, which theoretically accelerate the training in right direction and helps loss function not to jump over minimum.

Using following equation:

$$
\delta = c\cdot \delta_{old} + (1-c) \cdot \delta_{new}
$$

### Regularization

This project enables both L1 and L2 regularization method. They are controlled directly by the magnitude of the L1 and L2 coefficient.

## Performance

### CIFAR-10 Dataset

Source: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

Test accuracy: ~50%

### MNIST

Source: https://www.openml.org/search?type=data&status=active&id=554

Test accuracy: ~97%
