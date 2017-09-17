# Activation Classes
# For non-linearity in hidden and output layer
# Init with optional name
# Methods for forward pass and backward passes (derivatives), takes inputs


import numpy as np

# Logistic sigmoid
# Output and hidden rec.
class Sigmoid:
    def __init__(self, name="sigmoid"):
        self.name = name
        self.type= 'activation'
  
    def forward_pass(self, inputs):
        # 1 / ( 1 + e^(-z) )
        inputs = np.array(inputs)
        return 1 / (1 + np.exp(-inputs))

    def backward_pass(self, inputs):
        outputs = self.forward_pass(inputs)
        # Derivative wrt outputs of sigmoid, x: x * (1 - x)
        outputs = np.array(outputs)
        return outputs * (1 - outputs)

# Rectified linear unit
# Hidden rec.
class ReLU:
    def __init__(self, name="ReLU"):
        self.name = name
        self.type= 'activation'

    def forward_pass(self, inputs):
        # max{0, input}
        inputs = np.array(inputs)
        return np.maximum(inputs, 0)

    def backward_pass(self, nets):
        # Naive derivative wrt nets, ie inputs to ReLU activation, x: 0 if x <= 0 and 1 if x > 0
        nets = np.array(nets)
        rv = np.ones(nets.shape)
        rv[np.where(nets <= 0)[0]] = 0.0
        return rv

# Output only
# Assumes each input column is a class, and each row a training case
# Assumes one hot encoded labels
class Softmax:
    def __init__(self, name="Sigmoid"):
        self.name = name
        self.type= 'activation'
    def forward_pass(self, inputs):
        inputs = np.array(inputs, ndmin=2)
        return np.exp(inputs) / np.sum(np.exp(inputs), axis=1, keepdims=True)

    # Using the cross entropy softmax loss, the gradient is fully computed there
    # We just return ones for multiplication in back prop
    def backward_pass(self, inputs):
        inputs = np.array(inputs)
        return inputs
