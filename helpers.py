# Helpers
# Functions to get data into best format for inputs in deep nets

import numpy as np

# Add a column of ones to the front of the inputs for X0 = 1 feature
def add_bias(inputs):
    inputs = np.array(inputs, ndmin=2)
    return np.insert(inputs, 0, 1, axis=1) 

# For each feature in matrix, subtracts mean to create 0 mean, divides by std deviation to set std deviation to 1
# Returns the scale coefficients to get test/ other data on same scale
def standardize(inputs, scale=None):
    inputs = np.array(inputs, ndmin=2)
    if not scale:
        scale = np.max(inputs)
    return inputs / scale * 0.99, scale
    
# Given a graph and data, compute the prediction accuracy
def accuracy(inputs, labels, g):
    s = []
    y_ = g.forward_propagate(inputs, True)
    for i, j in zip(y_, labels):
        if np.argmax(i) == np.argmax(j):
            s.append(1)
        else:
            s.append(0)
    return sum(s)/len(s)
