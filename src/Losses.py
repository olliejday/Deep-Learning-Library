import numpy as np
from Core import Layer


class CrossEntropy(Layer):
    def __init__(self, size_l, name="cross entropy loss"):
        """
        Loss function for classification
        :param targets: the targets for the data
        """
        self.size_l = size_l
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


class CrossEntropySoftmax(Layer):
    def __init__(self, size_l, name="cross entropy loss"):
        """
        Loss function for classification
        :param targets: the targets for the data
        """
        self.size_l = size_l
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


class MeanSquaredError(Layer):
    def __init__(self, size_l, name="mean squared error loss"):
        """
        Loss function for regression
        :param targets: the targets for the data
        """
        self.size_l = size_l
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
