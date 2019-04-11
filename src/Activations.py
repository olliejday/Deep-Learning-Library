import numpy as np
from Core import Layer


class Activation(Layer):
    def __call__(self, node):
        """
        Overwrite call to setup based on previous layer.
        We set the size based on the last layer because an activation is element wise so doesn't change output shape
        :param node: the node to follow this layer in the model
        :return: this node
        """
        # call super layer setup
        Layer.__call__(self, node)
        self.size_l = self.previous.size_l
        return self


class ReLU(Activation):
    def __init__(self, name="relu"):
        """
        ReLU layer
        :param name: name of layer
        """
        self._type = "activation"
        self.name = name

    def __str__(self):
        return "{}".format(self.name)

    def forward(self, X):
        """
        Computes forward pass on ReLU activation
        max(x_i, 0)
        :param X: inputs (M, size_l)
        :return: outputs of ReLU (M, size_l)
        """
        return np.maximum(X, 0)

    def backward(self, x, g):
        """
        Computes gradients of outputs of ReLU with respect to inputs
        Gradient is x < 0 and 1 x > 0 (we set to 0 for x = 0)
        :param x: inputs of shape (size_l,)
        :param g: gradient delta of (size_l_prev,)
        :return: new gradient back propagated, (size_l,)
        """
        return np.squeeze(g * np.array(x > 0).astype(float))


class Softmax(Activation):
    def __init__(self, name="softmax"):
        """
        Softmax output layer
        :param name: name of layer
        """
        self._type = "activation"
        self.name = name

    def __str__(self):
        return "{}".format(self.name)

    def forward(self, X):
        """
        Computes forward pass on softmax activation
        :param X: inputs (M, size_l)
        :return: outputs of softmax (M, size_l)
        """
        exps = np.exp(X)
        return exps / np.sum(exps, axis=0, keepdims=True)

    def backward(self, x, g):
        """
        Just pass on the gradient, the actual gradient is computed in
        CrossEntropySoftmax for numerical stability
        :param x: inputs of shape (size_l,)
        :param g: gradient delta of (size_l_prev,)
        :return: new gradient back propagated, (size_l,)
        """
        return g
