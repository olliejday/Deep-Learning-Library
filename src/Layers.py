import numpy as np
from Core import Layer
from Initialisers import weight_init, bias_init

class Inputs:
    def __init__(self, size_l, name="inputs"):
        """
        Mostly a placeholder for inputs layer
        :param size_l: number of input units
        :param name: name of layer
        """
        self.size_l = size_l
        self.name = name
        self.next = None
        self._type = "inputs"

    def __str__(self):
        return "{} :: {}".format(self.name, self.size_l)

    def forward(self, X):
        """
        Forward pass on input just returns inputs if correct shape
        :param X: input batch (M, size_l)
        :return: the inputs
        """
        if X.shape[1] != self.size_l:
            raise Exception("Size of input {} must be {}".format(self.name, self.size_l))
        return X


class Dense(Layer):
    def __init__(self, size_l_prev, size_l, weights_initialiser="xavier_uniform", bias_initialiser="zeros",
                 name="dense"):
        """
        Init a Dense layer (WX + b)
        :param size_l_prev: size of prev layer (inputs)
        :param size_l: size of outputs
        :param weights_initialiser: weight initialiser function (default xavier_uniform)
        :param bias_initialiser: bias initialiser function (default zeros)
        :param name: name of layer
        """
        self.W = weight_init(weights_initialiser, size_l_prev, size_l)
        self.b = bias_init(bias_initialiser, size_l)
        self.size_l_prev = size_l_prev
        self.size_l = size_l
        self._type = "dense"
        self.name = name

    def __str__(self):
        return "{} :: {}, {}".format(self.name, self.size_l_prev, self.size_l)

    def forward(self, X):
        """
        Computes forward pass of linear layer on M batched examples
        :param X: Inputs of (M, size_l_prev)
        :return: Ouputs of size (M, size_l)
        """
        return np.dot(X, self.W) + self.b

    def gradient(self, x):
        """
        Computes gradients of outputs of linear layer on a single example with respect to the weights and biases
        :param x: input of (size_l_prev,)
        :return: Gradients with respect to weights size (size_l_prev, size_l) and biases size (size_l,)
        """
        return x

    def backward(self, x, g):
        """
        Computes gradients of outputs of linear layer on a single example with respect to the inputs
        :param x: input of (size_l_prev,)
        :param g: gradient delta of (size_l_prev,)
        :return: new gradient back propagated, (size_l,)
        """
        return np.dot(self.W, g)
