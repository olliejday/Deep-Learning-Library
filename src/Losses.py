import numpy as np
from Core import Layer


class Loss(Layer):
    def __call__(self, node):
        """
        Overwrite call to setup based on previous layer.
        We set the size based on the last layer because a loss doesn't change output shape
        :param node: the node to follow this layer in the model
        :return: this node
        """
        # call super layer setup
        Layer.__call__(self, node)
        self.size_l = self.previous.size_l
        return self


class CrossEntropy(Loss):
    def __init__(self, name="cross entropy loss"):
        """
        Loss function for classification
        """
        self.name = name

    def __str__(self):
        return "{} :: {}".format(self.name, self.size_l)

    def forward(self, logits, targets):
        """
        Computes forward pass on loss
        :param logits: (M, size_l)
        :param targets: (M, size_l) - one hot
        :return: outputs of loss (M, size_l)
        """
        # log likelihood where logits are seen as class probabilities
        # Y used as a mask on true values of X
        return - np.sum(np.log(logits) * targets) / self.size_l

    def gradient(self, logits, targets):
        """
        Computes gradients on loss with respect to logits
        :param logits: (size_l,)
        :param targets: (size_l,) - one hot
        :return: gradients of loss (size_l,)
        """
        return - targets / logits


class CrossEntropySoftmax(Loss):
    def __init__(self, name="cross entropy loss"):
        """
        Loss function for classification
        """
        self.name = name

    def __str__(self):
        return "{} :: {}".format(self.name, self.size_l)

    def forward(self, logits, targets):
        """
        Computes forward pass on loss
        :param logits: (M, size_l)
        :param targets: (M, size_l) - one hot
        :return: outputs of loss (M, size_l)
        """
        # log likelihood where logits are seen as class probabilities
        # Y used as a mask on true values of X
        return - np.sum(np.log(logits) * targets) / self.size_l

    def gradient(self, logits, targets):
        """
        Computes gradients on loss with respect to logits
        :param logits: (size_l,)
        :param targets: (size_l,) - one hot
        :return: gradients of loss (size_l, )
        """
        return np.dot(-np.eye(logits.shape[0]) + logits[None].T, targets)


class MeanSquaredError(Loss):
    def __init__(self, name="mean squared error loss"):
        """
        Loss function for regression
        """
        self.name = name

    def __str__(self):
        return "{} :: {}".format(self.name, self.size_l)

    def forward(self, logits, targets):
        """
        Computes forward pass on loss
        :param logits: (M, size_l)
        :param targets: (M, size_l)
        :return: outputs of loss (M, size_l)
        """
        return np.sum(-(logits - targets) ** 2) / targets.shape[0]

    def gradient(self, logits, targets):
        """
        Computes gradients on loss with respect to logits
        :param logits: (size_l,)
        :param targets: (size_l,) - one hot
        :return: gradients of loss (size_l,)
        """
        return logits - targets
