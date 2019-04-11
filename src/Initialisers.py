import numpy as np


def weight_init(weights_initialiser, size_l_prev, size_l):
    """
    Initialise a weight matrix
    :param weights_initialiser: type of initialiser to use {"xavier_uniform"}
    :param size_l_prev: size of previous layer
    :param size_l: size of current layer
    :return: weights matrix shape [size_l_prev, size_l]
    """
    if weights_initialiser == "xavier_uniform":
        limit = np.sqrt(6 / float(size_l_prev + size_l))
        return np.random.uniform(-limit, limit, (size_l_prev, size_l))


def bias_init(bias_initialiser, size_l):
    """
    Initialise a bias vector
    :param bias_initialiser: type of initialiser to use {"zeros"}
    :param size_l: size of bias vector
    :return: a [size_l,] bias vector
    """
    if bias_initialiser == "zeros":
        return np.zeros(size_l)
