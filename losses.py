# Cost/Error/Loss Function Classes
# Init with optional name
# impelements forward pass and backward pass (derivatives) which both take inputs and labels

import numpy as np

# Cost fn's back pass takes labels as well as inputs
class MSE: # Mean squared error (technically a misnomer as it's 1/2 not 1/n)
    def __init__(self, name="mean_sqd_error"):
        self.name = name
        self.type= 'loss'
    def forward_pass(self, outputs, labels):
        # (1/2) * sum over j (target_j - output_j)^2
        # In matrix form MSE = diagonal of [ (1/2) * ((TARGETS - OUTPUTS)T * (TARGETS - OUTPUTS)) ]
        labels = np.array(labels)
        outputs = np.array(outputs)
        if labels.shape != outputs.shape:
            print("\nError in MSE, outputs {} and labels {} not same shape\n".format(outputs.shape, labels.shape))
            return
        # Return line computes the MSE with matrix dot products then scales by the 1/2
        # then extracts the resulting diagonal for the MSE vector
        return np.array([(1/2) * np.dot((labels - outputs).T, (labels - outputs))[i,i] for i in range(outputs.shape[1])])

    def backward_pass(self, outputs, labels):
        # Derivative of MSE wrt the outputs of the model
        labels = np.array(labels)
        outputs = np.array(outputs)
        if labels.shape != outputs.shape:
            print("\nError in MSE, outputs {} and labels {} not same shape\n".format(outputs.shape, labels.shape))
            return
        return (outputs - labels)

# Cost fn's back pass takes labels as well as inputs
# Recommended for use with softmax, and written only for use with softmax, bc the derivative applies to softmax
class CrossEntropySoftmax:
    def __init__(self, name="cross_entropy"):
        self.name = name
        self.type= 'loss'
    def forward_pass(self, outputs, labels):
        outputs = np.array(outputs)
        y = np.array(labels)
        labels = np.argmax(labels, axis=1)
        num_ex = outputs.shape[0]
        # Element times the labels and outputs will zero all non answer classes, sum to get into vector and log
        log_probs = -np.log(outputs[range(num_ex), labels])
        return np.sum(log_probs) / num_ex
        
    # Derivative of cross entropy (for a softmax) is p_k - 1[y_i = k] which can be implemented by subtracting one hot labels from the outputs of softmax
    # Assumes one hot encoded labels, which are then converted back into integer labels
    def backward_pass(self, outputs, labels):
        labels = np.argmax(labels, axis=1)
        outputs = np.array(outputs)
        outputs[range(outputs.shape[0]), labels] -= 1
        outputs /= outputs.shape[0]
        return outputs
