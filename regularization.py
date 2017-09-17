# Regularization classes
# These wrap onto the linear class, which contains the weights to be regularized
# ZeroReg() is the default regularization and returns zero both back and forw passes
# but it does give a good template for how regularization is implemented
# In general, regularization should be insantiated with a lambda term that defines the 
# weight of regularization term in the total cost, this is called '.weight' as lambda is a key word
# Also init with linear, which is the linear object to wrap to, this is passed by passing self in the Linear class's add_regularization()
# Objects are instantiated within the Linear class's add_regularization method
# This is bc the init must take, via self, a Linear object to wrap to

import numpy as np

class ZeroReg:
    def __init__(self, linear, reg_weight=0):
        self.linear = linear # Wrap to a linear node
        self.reg_weight =reg_weight
        self.name = 'zero_regularization' # Default regularization must be named this for identifying
        self.type = 'regularization'

    def forward_pass(self):
        return 0

    def backward_pass(self):
        return 0 

# Simple weight decay regularization
# Reg = 1/2 * lambda * weight^2
# Derivative wrt weights = lambda * weight
class WeightDecay:
    def __init__(self, linear, reg_weight=1e-3):
        self.linear = linear # Wrap to a linear node
        self.reg_weight = reg_weight
        self.name = "weight_decay"
        self.type = 'regularization'

    def forward_pass(self):
        return 0.5 * self.reg_weight * np.sum(self.linear.weights * self.linear.weights)

    def backward_pass(self):
        return self.reg_weight * self.linear.weights 
