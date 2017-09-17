# Feed-Forward-Neural-Network-Library

A modular Feedforward Network library in Python, written using Numpy. A basic set of affine layers and non-linear activations, regularisations, optimisation algorithms and cost functions can be combined in a fairly flexible order to produce a FFNN.

#### Current options

ACTIVATIONS: ReLU, Softmax, Sigmoid
LOSS: Mean Squared Error (MSE), Cross Entropy (for Softmax)
OPTIMISERS: ADAM, Gradient Descent
REGULARISATION: Weight Decay
Planned additions: Dropout

#### DOCUMENTATION

Init a graph object by ```g = Graph()```

Add nodes with the ```g.add_node()``` method

Layers can be added in the following patterns, described in ```add_node```:

Start with a ```Linear()``` node

Then an activation node eg. ```ReLU()``` or ```Sigmoid()```


To add regularization, wrap the regularization op to the linear node. For example, the following syntax after adding a linear node will add a ```WeightDecay``` to that unit:
	
	```g.graph[-1].add_regularization(WeightDecay, **kwargs) ```

where ```**kwargs``` can include name, lambda weight etc. Note the use of the class ```WeightDecay``` without brackets as we instantiate the regularization class within linear in order to pass it self and wrap to the linear object.

Add a final layer with a ```Linear()``` and output activation eg. ```Sigmoid()``` or ```Softmax()```

Finally a loss operation eg. ```MSE()``` or ```CrossEntropySoftmax()```

To train add an optimizer eg. ```opt = GradientDescent()```

Then call ```opt.update(inputs, labels, graph)```

Finally to run on test output simply call ```graph.forward_propagate(inputs, forward_only=True)```


The library works by building up a list which acts as a computation graph. All operations have forward and backward pass operations which can be used in forward or backwards order to compute outputs or gradients respectively.

#### EXAMPLE USAGE

(See commented blocks in feed_forward_nn.py for code and to run)

INSERT PICS OF TRAINING GRAPH HERE

#### DEPENDENCIES
Numpy

Matplotlib is optional for plots in example code

#### Adding CNN and RNN soon. 

