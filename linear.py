# Linear computation class
# Similar to activation classes,
# Init with shape of weights, which must match with the network, from which biaes are also created
# Implements a weight update method which is passed the update matrix (with the learning rate already producted)
# Regularization is a class property which is a class as defined in regularization.py, defaults to the zero_reg.
# which returns zeros both ways so no regularization

import numpy as np

from regularization import ZeroReg

class Linear:  
    def __init__(self, 
                shape, 
                name='linear', 
                weight_init= lambda x: 0.01 * np.random.randn(x[0], x[1]),
                bias_init= lambda x: 0.01 * np.zeros((1, x)),
                no_bias=False):
        self.type = 'linear'
        self.name = name
        self.shape = shape
        self.weights = weight_init(shape)
        self.biases = bias_init(shape[1])
        self.no_bias = no_bias
        self.regularization = ZeroReg(self)

    def forward_pass(self, inputs):
        # Matrix dot product of design matrices W * X
        inputs = np.array(inputs)
        if self.no_bias: return np.dot(inputs, self.weights)
        return np.dot(inputs, self.weights) + self.biases

    def backward_pass(self, inputs, wrt='inputs'):
        # Linear = sum [w_i * x_i]
        # Derivative wrt weights is just the inputs: x_i
        # Derivative wrt inputs is just the weights: w_i
        if wrt.lower() == 'weights':
            return inputs
        return self.weights
        
    def update_weights(self, update_weights, update_biases):
        if np.array(update_weights).shape != self.weights.shape:
            print("\nCannot update matrix {} for weights size {}\n".format(np.array(update).shape, self.weights.shape))
            return
        self.weights -= update_weights
        self.biases -= np.sum(update_biases, axis=0, keepdims=True)

   
    # To add regularization you pass the regularization.py CLASS, not the object
    # This is bc the init takes the linear object to wrap to, passed below in self
    # Also takes a *kwargs which can update name and lambda weight etc. which is wrapped 
    # in try... except to catch problems there
    def add_regularization(self, Class, **kwargs):
        # Init the regularization class with the kwargs and the linear object - self
        # Catch TypeError on the regularization init arguments
        try:
            node = Class(self, **kwargs)
        except TypeError:
            print("\nCould not add regularization with arguments")
            print(kwargs)
            return
        # Few book keeping checks
        if node.type != "regularization":
            print("\nCannot regularize with node of type {}, name {}\n".format(node.type, node.name))
            return
        if self.regularization.name != 'zero_regularization': # If updated the reg for this node already
            print("\nNode already has regularization: name {}\n".format(self.regularization.name))
            return
       
        self.regularization = node 
