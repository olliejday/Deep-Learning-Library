# Learning Procedure Classes

import numpy as np

# Given a graph, organizes the back propagation and perform the update steps
# Init with learning rate
class GradientDescent:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        
    def update(self, inputs, labels, graph, return_cost=False):
        # Back prop to get gradients: partial errror function_p / partial Weight_pj for case p and layer j
        if return_cost:
            weights_deltas, bias_deltas, cost = graph.back_propagate(inputs, labels, return_cost)
        else:
            weights_deltas, bias_deltas = graph.back_propagate(inputs, labels)
        i = 1 # Index into gradients, which are reversed compared to graph
        for node in graph.graph:
            if node.type == 'linear':
                weight_update = weights_deltas[-i] * self.learning_rate
                bias_update = bias_deltas[-i] * self.learning_rate
                node.update_weights(weight_update, bias_update)
                i += 1
        if return_cost:
            return cost
            
class ADAM:
    
    # Init to set all the params for ADAM optimizer, also needs an instance of the graph, just for sizing v and s
    def __init__(self, beta1, beta2, epsilon, learning_rate, graph):
        self.b1 = beta1
        self.b2 = beta2
        self.epsilon = epsilon
        self.lr = learning_rate
        # v is moving avg of gradients, s is moving avg of squared gradients.
        self.v, self.s = [], [] # Both v and s of form [ ['weights': w1, 'bias': b1], ['weights': w2, 'bias': b2] ...]
        for node in graph.graph:
            if node.type == 'linear':
                self.v.append({'weights': np.zeros(node.shape), 'bias': np.zeros(node.shape[1])})
                self.s.append({'weights': np.zeros(node.shape), 'bias': np.zeros(node.shape[1])})
        self.t = 1 # Number of steps taken with this update class, start on 1 to avoid division by zero

    def update(self, inputs, labels, graph, return_cost=False):
        # Back prop to get gradients: partial errror function_p / partial Weight_pj for case p and layer j
        if return_cost:
            weights_deltas, bias_deltas, cost = graph.back_propagate(inputs, labels, return_cost)
        else:
            weights_deltas, bias_deltas = graph.back_propagate(inputs, labels)
        
        i = 1 # Index into gradients, which are reversed compared to graph
              # And i indexes into v and s above which are same as graph's linear node ordering
        
        # Copy shapes of v and s into bias-corrected copies
        v_corrected = [i for i in self.v] 
        s_corrected = [i for i in self.s]
        for node in graph.graph:
            if node.type == 'linear':
                # Update v
                self.v[i - 1]['weights'] = self.b1 * self.v[i - 1]['weights'] + (1 - self.b1) * weights_deltas[-i]
                self.v[i - 1]['bias'] = self.b1 * self.v[i - 1]['bias'] + (1 - self.b1) * bias_deltas[-i]
                # Compute bias-corrected
                v_corrected_weights = self.v[i - 1]['weights'] / (1 - (self.b1 ** self.t))
                v_corrected_bias = self.v[i - 1]['bias'] / (1 - (self.b1 ** self.t))

                # Update s
                self.s[i - 1]['weights'] = self.b2 * self.s[i - 1]['weights'] + (1 - self.b2) * (weights_deltas[-i] ** 2)
                self.s[i - 1]['bias'] = self.b2 * self.s[i - 1]['bias'] + (1 - self.b2) * (bias_deltas[-i] ** 2)
                # Bias correct s
                s_corrected_weights = self.s[i - 1]['weights'] / (1 - (self.b2 ** self.t))
                s_corrected_bias = self.s[i - 1]['bias'] / (1 - (self.b2 ** self.t))
                
                # Update parameters
                weight_update = self.lr * (v_corrected_weights / (np.sqrt(s_corrected_weights + self.epsilon)))
                bias_update = self.lr * (v_corrected_bias / (np.sqrt(s_corrected_bias + self.epsilon)))
                node.update_weights(weight_update, bias_update)
                
                i += 1

        self.t += 1 # Update the number steps 
                
        if return_cost:
            return cost
        
        
