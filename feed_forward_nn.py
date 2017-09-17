# Implements a basic feed forward neural network on a computational graph.

# TO-DO:
# Docstrings on functions
# Dropout
# RNNs and CNNs -> perhaps as separate libraries?

# Syntax:
# Init a graph object by g = Graph()
# Add nodes with the g.add_node() method
# Nodes must follow the following patterns, described in add_node:
# Start with a Linear() node
# Then an activation node eg. ReLU() or Sigmoid()
# Then however many hidden layers of the form Linear() then activation
# To add regularization, wrap the regularization op to the linear node
# the following syntax after adding a linear node will get that unit and add a WeightDecay
# regularization: g.graph[-1].add_regularization(WeightDecay, **kwargs) 
# where **kwargs can include name, lambda weight etc. Note the use of the class WeightDecay without brackets
# as we instantiate the regularization class within linear in order to pass it self and wrap
# to the linear object.
# Then a final Linear() and output activation eg. Sigmoid() or Softmax()
# Finally a loss operation eg. MSE() or CrossEntropySoftmax()
# To train add an optimizer eg. opt = GradientDescent()
# Then call opt.update(inputs, labels, graph)
# Finally to run on test output simply call graph.forward_propagate(inputs, forward_only=True)

import numpy as np
from random import sample
import matplotlib.pyplot as plt # Optional, for plotting
import matplotlib.gridspec as gridspec # Optional, for plotting

from activations import ReLU, Sigmoid, Softmax
from linear import Linear
from losses import MSE, CrossEntropySoftmax
from optimizers import GradientDescent, ADAM
from helpers import add_bias, standardize, accuracy
from regularization import WeightDecay

# Defines a computation graph structure with forward and backward propagation for a 
# feed forward network - Fully connected ie. no skipped connections
class Graph:
    # 
    def __init__(self):
        self.graph = []
    
    def print_graph(self):
        print("\n{:<18}{:<18}{:<18}{:<18}{:<18}{:<10}".format("NAME", "TYPE", "WEIGHT (linear)", "BIAS (linear)", "REGULARIZATION", "Reg. Lambda"))
        
        for node in self.graph:
            if node.type == 'linear':
                print("{:<18}{:<18}{:<18}{:<18}{:<18}{:<10}".format(node.name, node.type, str(node.shape), '['+str(node.shape[1])+']', node.regularization.name + ',', str(node.regularization.reg_weight)))
            else:
                print("{:<18}{:<18}".format(node.name, node.type))
        print()

    def add_node(self, node):
        if not node or hasattr(node, 'bad_init'): # Catch dodgy nodes
            print("\nNode not well defined.\n")
            return
        
        if node.type == 'regularization':
            print("\nRegularization must be wrapped to a linear node.\n")
            return

        if len(self.graph) > 0:
            # Don't support same node twice in a row ie. must alternate linear and activation
            if self.graph[-1].type == node.type:
                print("\nCannot order {} node in graph after {} node\n".format(node.type, self.graph[-1].type))
                self.print_graph()
                return
    
            # Loss must be last operation
            if self.graph[-1].type == "loss":
                print("\nCannot define operations after loss\n")
                self.print_graph()
                return

            # Consecutive linear layers must be same size
            if node.type == 'linear' and 'linear' in [node.type for node in self.graph]:
                last_linear = None
                for n in self.graph:
                    if n.type == 'linear':
                        last_linear = n

                if node.shape[0] != last_linear.shape[1]:
                        print("\nIncompatible sizings, last linear shape {} does not fit with new linear layer shape {}\n".format(node.shape[0], last_linear.shape[1]))
                        return

        else: # if first layer, must be linear
            if node.type != 'linear':
                print("\nFirst layer must be linear\n")
                return
        
        # Only allow one loss operation
        if node.type == "loss" and "loss" in [node.type for node in self.graph]:
            print("\nCannot have two loss operations defined.\n")
            self.print_graph()
            return

        # All good, add the node
        else:
            self.graph.append(node)
        
    def check_graph(self, inputs=None, labels=None):
        # Ensure graph defined
        if not self.graph or self.graph == []:
            print("\nGraph not defined\n")
            self.print_graph()
            return False
        # Make sure first node is linear and penultimate node i linear
        if not self.graph[0].type == 'linear':
            print("\nFirst unit must be linear\n")
            self.print_graph()
            return False
        if not self.graph[-3].type == 'linear':
            print("\nPenultimate unit (unit before final activation and loss) must be linear\n")
            self.print_graph()
            return False
        
        if inputs is not None:
            # Ensure inputs correct shape 
            if inputs.shape[1] != self.graph[0].shape[0]:
                print("\nCannot propagate inputs of shape {}, network requires {} inputs.\n".format(inputs.shape, self.graph[0].shape))
                self.print_graph()
                return False
        if labels is not None:
            # Ensure loss defined
            if self.graph[-1].type != "loss":
                print("\nLoss not deinfed.\n")
                self.print_graph()
                return False
            # Ensure labels correct shape 
            if labels.shape[1] != self.graph[-3].shape[1]:
                print("\nCannot propagate labels of shape {}, network requires {} outputs.\n".format(labels.shape, self.graph[-3].shape))
                self.print_graph()
                return False

        return True # All good

    # Propagates inputs of shape [# examples, # features] throught the network
    # Assumes bias fetures already in inputs (helpers.add_bias)
    # Best to standardize inputs (helpers.standardize)
    # Forward only at test time, returns final output activations only, otherwise
    # activations of all layers are returned
    # Returns the model's outputs for every node
    def forward_propagate(self, inputs, forward_only=False):
        act = np.array(inputs, ndmin=2) # Initial activations
        if not forward_only: 
            activations = [act] # Store all activations, starting with inputs
        # Ensure graph defined
        if not self.check_graph(act): return
        
        # Pass through each layer
        for node in self.graph: # All layers except output layer
            # break on loss for forward passes
            if node.type == "loss":
                break  
            act = node.forward_pass(act)
            # Store all activations, but only after activation functions (so not linear nodes)
            if not forward_only: activations.append(act)
        
        # Compute and return output layer
        if forward_only: return act
        return activations
    
    # Returns the highest probability class from our model for given inputs
    def classify(self, inputs):
        return np.argmax(self.forward_propagate(inputs, True), axis=1)

    # Back propagates to compute the gradients
    # Assumes one-hot encoded labels equivalent to outputs, local variables:
    # gradients/ updates: partial errror function_p / partial Weight_pj for case p and layer j
    # deltas: partial error function_p / partial net_pj (W * X before activation) for case p and layer j
    # returns gradients and optionally cost in [unreg., regularized]
    def back_propagate(self, inputs, labels, return_cost=False):
        # Get inputs and labels into numpy arrays
        inputs = np.array(inputs, ndmin=2)
        labels = np.array(labels, ndmin=2)
        m = len(inputs)
        
        # Ensure graph and loss defined
        if not self.check_graph(inputs, labels): return

        # Forward pass to get layer activations
        activations = self.forward_propagate(inputs)
        
        if return_cost:
            rv = [np.mean(self.graph[-1].forward_pass(activations[-1], labels))] # last activations are outputs, cost
            reg_term = 0
            for node in self.graph:
                if node.type == 'Linear':
                    if node.regularization.name == 'weight_decay':
                        reg_term += node.regularization.forward_pass() / (node.regularization.reg_weight / (2*len(inputs)))  
            rv.append(rv[0] + reg_term) # Regularize

        weight_deltas = [] # to store computed gradients
        bias_deltas = []

        # Output derivatives
        # Multiplied by output layer activation in below loop
        # Last layer is loss, compute the loss derivative
        # Actually cost delta is 1 (dL/dL = 1), but use this as a proxy for dL/da for final activation, second term is da/dz
        delta = self.graph[-1].backward_pass(activations[-1], labels) * self.graph[-2].backward_pass(activations[-2])
        update = (np.dot(activations[-3].T, delta) / m) + self.graph[-3].regularization.backward_pass() # delta and regularization term
        weight_deltas.append(update)
        bias_deltas.append(delta / m)
        
        for i in range(3, len(activations), 2):
            delta = np.dot(delta, self.graph[-i].backward_pass(activations[-i], 'inputs').T) * self.graph[-i-1].backward_pass(activations[-i-1])
            update = (np.dot(activations[-i-2].T, delta) / m) + self.graph[-i-2].regularization.backward_pass() # delta and regularization term (reg applies to previous layer before)
            weight_deltas.append(update)
            bias_deltas.append(delta / m)
        
        if return_cost: return weight_deltas, bias_deltas, rv
        return weight_deltas, bias_deltas


# MNIST example
# Regularisation
# Sigmoid and MSE
'''
# Read in data
with open('mnist_train_100.csv') as f:
    data = f.readlines()
all_values = np.asfarray([line.split(',') for line in data])

X = (all_values[:,1:] / 255.0 * 0.98) + 0.01 # Scale inputs for sigmoid activation's best (0,1) range
labels = np.array(all_values[:,0], dtype=np.int32) # First number is label
y = np.zeros((len(labels), 10)) + 0.01 # One hot encode within (0,1) range for sigmoid
y[range(len(labels)), labels] = 0.99

testX = X[:10] # Small set to print out
testY = y[:10]

print("\nTraining will small MNIST subset")
print("X: ", X.shape)
print("y: ", y.shape)

# Define network graph
g = Graph()
g.add_node(Linear((784, 100)))
g.graph[-1].add_regularization(WeightDecay)
g.add_node(Sigmoid())
g.add_node(Linear((100, 10)))
g.graph[-1].add_regularization(WeightDecay)
g.add_node(Sigmoid())
g.add_node(MSE())

g.print_graph()

# Define optimizer
opt = GradientDescent(0.2)

# Initial test
print("\nInitial estimates")
print(g.classify(testX))
print("Labels")
print(np.argmax(testY,axis=1))
print()
# Train
cost_plot = [] # For plotting costs
acc_plot = [] # For plotting accuracy
plot_iters = [] # for x axis plots
for i in range(2000):
    cost_plot.append(opt.update(X, y, g, True)[1]) # Append regularized cost
    acc_plot.append(accuracy(X, y, g))
    plot_iters.append(i)
    if i % 500==0:
        print("{} training cost {}, accuracy {}".format(i, cost_plot[-1], acc_plot[-1]))
        
print("\nEstimates after training")
guesses = g.classify(testX)
print(guesses)
print("Labels")
print(np.argmax(testY,axis=1))
accuracy(X, y, g)

# Plots
fig = plt.figure(1)
gridspec.GridSpec(5,5)
plt.subplot2grid((5,5),(0,2),colspan=2,rowspan=2)
plt.title('Accuracy')
plt.plot(plot_iters,acc_plot)

plt.subplot2grid((5,5),(0,0),colspan=2,rowspan=2)
plt.title('Cost')
plt.plot(plot_iters,cost_plot)

for i in range(10):
    plt.subplot2grid((5,5),(3+(i%2),i%5))
    plt.title("Model output {}".format(guesses[i]))
    plt.imshow(np.reshape(testX[i],(28,28)), cmap="Greys")

fig.tight_layout()
fig.set_size_inches(w=11,h=7)
fig.show()

input() # Hold graph
'''

# XOR example
# ADAM optimizer vs SGD
# ReLU, Sigmoid with MSE cost
'''
X = np.array([[.1, .9],[.9,.1],[.1,.1],[.9,.9]])
# XOR labels
labels = np.array([[0.1,0.9], [0.1,0.9], [0.9, 0.1], [0.9, 0.1]])
print("\nINPUTS\n")
print(X)
print("\nLABELS\n")
print(np.argmax(labels, axis=1))

# SGD

# Define graph
g = Graph()
g.add_node(Linear((2, 50), no_bias=True))
g.add_node(ReLU())
g.add_node(Linear((50, 2), no_bias=True))
g.add_node(Softmax())
g.add_node(CrossEntropySoftmax())

g.print_graph()


print("\nGradient Descent\n")

# Define optimizer
opt = GradientDescent(30.0)
# Train
cost_plot = [] # For plotting costs
plot_iters = [] # for x axis plots
for i in range(10000):
    cost_plot.append(opt.update(X, labels, g, True)[1]) # Append regularized cost
    plot_iters.append(i)

# Plot training
plt.plot(plot_iters, cost_plot, label="Gradient Descent")
# Test
print(g.classify(X))

# ADAM
print("\nADAM\n")
# Define graph
g = Graph()
g.add_node(Linear((2, 50), no_bias=True))
g.add_node(ReLU())
g.add_node(Linear((50, 2), no_bias=True))
g.add_node(Sigmoid())
g.add_node(MSE())

# Define optimizer
opt = ADAM(0.9, 0.99, 1e-8, 0.1, g)
# Train
cost_plot = [] # For plotting costs
plot_iters = [] # for x axis plots
for i in range(10000):
    cost_plot.append(opt.update(X, labels, g, True)[1]) # Append regularized cost
    plot_iters.append(i)
    
# Plot training
plt.plot(plot_iters, cost_plot, label="ADAM")
# Test
print(g.classify(X))

plt.xlabel("Number of Iterations")
plt.ylabel("Error (MSE)")
plt.legend(loc='best')
plt.show()
input() # Hold graph
'''
