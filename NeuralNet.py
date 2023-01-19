# dependicies
import numpy as np
import cupy as cp
import Util

class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        Initialize activation type and placeholders.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU","output"]:  
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This can be used for computing gradients.
        self.x = None

    def __call__(self, z):
        """
        This method allows your instances to be callable.
        """
        return self.forward(z)

    def forward(self, z):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(z)

        elif self.activation_type == "tanh":
            return self.tanh(z)

        elif self.activation_type == "ReLU":
            return self.ReLU(z)

        elif self.activation_type == "output":
            return self.output(z)

    def derivative(self, z):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            return self.derivative_sigmoid(z)

        elif self.activation_type == "tanh":
            return self.derivative_tanh(z)

        elif self.activation_type == "ReLU":
            return self.derivative_ReLU(z)

        elif self.activation_type == "output":
            return self.derivative_output(z)


    def sigmoid(self, x):
        """
        The sigmoid activation function.
        """
        return 1 / (1 + cp.exp(-x))

    def tanh(self, x):
        """
        The tanh activation function.
        """
        return cp.tanh(x)

    def ReLU(self, x):
        """
        The ReLU activation function.
        """
        return cp.maximum(0, x)

    def output(self, x, epsilon=1e-8):
        """
        Output softmax Function deal with overflow.
        """
        # for numerical stability
        temp = cp.exp(x - cp.max(x))
        return cp.divide(temp, temp.sum(axis=1).reshape(-1, 1))


    def derivative_sigmoid(self,x):
        """
        The derivative of sigmoid function.
        """
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def derivative_tanh(self,x):
        """
        The derivative of tanh function.
        """
        return 1 - self.tanh(x)**2

    def derivative_ReLU(self,x):
        """
        The derivative of ReLU function.
        """
        return (x > 0) * 1
        
    def derivative_output(self, x):
        """
        Deliberately returning 1 for output layer case since we don't multiply by any activation for final layer's delta.
        """
        return x*0 + 1

class Regularization():
    """
    The class implements different types of regularization functions for
    your neural network layers.
    """
    def __init__(self, reg_coe):
        """
        Initialize regularization type and placeholders here.
        """
        self.L1_constant = reg_coe[0]
        self.L2_constant = reg_coe[1]

    # def reg_weight(self, w):
    #     """
    #     Compute the forward pass.
    #     """
    #     if self.regularization_type == "l2":
    #         return self.l2(w)

    #     elif self.regularization_type == "l1":
    #         return self.l1(w)

    def gradient(self, w):
        """
        Compute the regularization part of gradient
        """
        return self.L1_constant*self.l1(w) + self.L2_constant*self.l2(2)

    def l2(self, w):
        """
        The L2 regularization.
        """
        return w * 2

    def l1(self, w):
        """
        The L1 regularization.
        """
        return (w > 0) * 1.0


class Layer():
    """
    This class implements Fully Connected layers for your neural network.
    """

    def __init__(self, in_units, out_units, activation):
        """
        Define the architecture and create placeholders.
        """
        cp.random.seed(42)

        self.w = 0.01 * cp.random.random((in_units+1, out_units)) 
            # randomly assign the weight
            
        self.x = None                           # Save the input to forward in this
        self.a = None                           # output without activation
        self.z = None                           # Output After Activation
        self.delta = None                       # store the delta for back pass
        self.gradient = None                    # store the gradient without learning rate
        self.activation = activation            # Activation function
        self.acc_gradient = self.w * 0          # initilize the accumulated gradient term
        self.out_units = out_units

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def clear_cache(self):
        """
        Clear all the parameters except weights to prevent any information being wrongly used.
        """
        self.x = None
        self.a = None
        self.z = None
        self.delta = None
        # self.acc_gradient = self.w * 0

    def forward(self, x):
        """
        Compute the forward pass (activation of the weighted input) through the layer here and return it.
        """
        self.x = Util.append_bias(x)                # store the last unit's output (n * in_units)
        self.a = self.x @ self.w                    # store the output without activation (n * out_units)
        self.z = self.activation.forward(self.a)    # store the output with activation (n * out_units)

        return self.z

    def backward(self, deltaAndW):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input and
        computes gradient for its weights and the delta to pass to its previous layers. gradReqd 
        is used to specify whether to update the weights i.e. whether self.w should be updated after 
        calculating self.dw
        The delta expression (that you prove in PA1 part1) for any layer consists of delta and weights 
        from the next layer and derivative of the activation function of weighted inputs i.e. g'(a) of 
        that layer. Hence deltaCur (the input parameter) will have to be multiplied with the derivative 
        of the activation function of the weighted input of the current layer to actually get the delta 
        for the current layer. Remember, this is just one way of interpreting it and you are free to 
        interpret it any other way. Feel free to change the function signature if you think of an 
        alternative way to implement the delta calculation or the backward pass
        """
        # the deltaAndW term is the delta of this layer without multipling the derivative
        # self.delta = self.activation.derivative(self.a) * deltaAndW
        self.delta = self.activation.derivative(self.a) * deltaAndW[:, :self.out_units]
            # self delta should be n * out_units

        # it is important to notice that the delta of the bias node of this layer will not move backward
        # but its delta has to be changed based on the thing from next layer

        deltaAndWLast = self.delta @ self.w.T
            # notice we don't send the delta of bias units back since they don't have any weights
        return deltaAndWLast

    def calculate_gradient(self, learning_rate, z_last, delta, regularization):
        """
        Calculate the gradient of this layer.
        """
        # we should result in a in * out matrix

        batch_size = z_last.shape[0]
        self.gradient = -((z_last.T @ delta)/(batch_size) + regularization.gradient(self.w))

        return (-1) * learning_rate * self.gradient

    def momentum_gradient(self, gamma, gradient, record=True):
        """
        Calculate the momentum accumulated gradient.
        """
        acc_gradient = gamma * self.acc_gradient + (1-gamma) * gradient  # notice we provides a 1-gamma term
        if record:
            self.acc_gradient = acc_gradient
        return acc_gradient

    def update_weight(self, gradient):
        """
        This is the function to update one layer's weight based on gradient.
        """
        self.w = self.w + gradient

class MLPClassifier():
    """
    Create a Neural Network specified by some configuration.
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []                                    # Store all layers in this list.
        self.num_layers = len(config.layer_specs) - 1    # Set num layers here
        self.x = None                                       # Save the input to forward in this
        self.y = None                                       # For saving the output vector of the model
        self.targets = None                                 # For saving the targets
        self.learning_rate = config.learning_rate
        self.momentum = config.momentum
        self.momentum_gamma = config.momentum_gamma                      # the momentum gamma coefficient
        self.regularization = Regularization([config.L1_constant, config.L2_constant])                              # initilize the regularization
        self.reg_coe = [config.L1_constant, config.L2_constant]       # the regularization coefficient

        # Add layers specified by layer_specs.
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                self.layers.append(
                    Layer(config.layer_specs[i], config.layer_specs[i + 1], Activation(config.activation))
                    )
                        
            elif i == self.num_layers - 1:
                self.layers.append(
                    Layer(config.layer_specs[i], config.layer_specs[i + 1], Activation("output"))
                    )

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def clear_cache(self):
        for layer in  self.layers:
            layer.clear_cache()
        self.x = None
        self.y = None    

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return the loss.
        If targets are provided, return loss and accuracy/number of correct predictions as well.
        """
        self.x = x # record the input

        z = x 
        for layer in self.layers:
            # move forward with new outputs as next layer's input
            z = layer.forward(z)

        self.y = z # record the final output

        if targets is not None:
            return self.loss(z, targets=targets)

        return

    def loss(self, logits, targets, epsilon=1e-5):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        return -1 * np.sum(targets * np.log(logits+epsilon)) / (logits.shape[0])

    def backward(self, targets, gradReqd=True):
        '''
        Implement backpropagation here by calling backward method of Layers class.
        Call backward methods of individual layers.
        '''
        # load some hyperparameters
        learning_rate = self.learning_rate

        backward_layers = self.layers[::-1]

        # the first delta and w result for the output layer
        deltaAndW = targets - self.y

        for layer in backward_layers:
            # this method store one layer's delta and create new deltaAndW
            # notice the deltaAndW we left for next layer is based on old weights!
            deltaAndW = layer.backward(deltaAndW) 

            # last layer's z
            z_last = layer.x 

            # calculate the layer gradient 
            layer_gradient = layer.calculate_gradient(learning_rate, z_last, layer.delta, self.regularization)            
            
            if self.momentum:
                # if use momentum method
                layer_gradient = layer.momentum_gradient(self.momentum_gamma, layer_gradient)

            if gradReqd:
                # if required to update the weight
                layer.update_weight(layer_gradient)