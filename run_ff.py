import numpy as np
import matplotlib.pyplot as plt

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


class Inputs:
    def __init__(self, size_l, name="inputs"):
        """
        Mostly a placeholder for inputs layer
        :param size_l: number of input units
        :param name: name of layer
        """
        self.size_l = size_l
        self.name = name
        self.next = None
        self._type = "inputs"

    def __str__(self):
        return "{} :: {}".format(self.name, self.size_l)

    def forward(self, X):
        """
        Forward pass on input just returns inputs if correct shape
        :param X: input batch (M, size_l)
        :return: the inputs
        """
        if X.shape[1] != self.size_l:
            raise Exception("Size of input {} must be {}".format(self.name, self.size_l))
        return X


class Layer:
    def __init__(self):
        self.next = None
        self.previous = None

    def __call__(self, node):
        """
        Constructs the graph with node as the child of this node.
        :param node: node to be child
        :return: the layer object
        """
        # if we need to check sizes
        if hasattr(node, "size_l") and hasattr(self, "size_l_prev"):
            if node.size_l != self.size_l_prev:
                raise Exception("Previous layer {} outputs size {} but layer {} input size {}".format(node.name,
                                                                                                      node.size_l,
                                                                                                      self.name,
                                                                                                      self.size_l_prev))
        # build the graph
        self.previous = node
        node.next = self
        return self


class Dense(Layer):
    def __init__(self, size_l_prev, size_l, weights_initialiser="xavier_uniform", bias_initialiser="zeros",
                 name="dense"):
        """
        Init a Dense layer (WX + b)
        :param size_l_prev: size of prev layer (inputs)
        :param size_l: size of outputs
        :param weights_initialiser: weight initialiser function (default xavier_uniform)
        :param bias_initialiser: bias initialiser function (default zeros)
        :param name: name of layer
        """
        self.W = weight_init(weights_initialiser, size_l_prev, size_l)
        self.b = bias_init(bias_initialiser, size_l)
        self.size_l_prev = size_l_prev
        self.size_l = size_l
        self._type = "dense"
        self.name = name

    def __str__(self):
        return "{} :: {}, {}".format(self.name, self.size_l_prev, self.size_l)

    def forward(self, X):
        """
        Computes forward pass of linear layer on M batched examples
        :param X: Inputs of (M, size_l_prev)
        :return: Ouputs of size (M, size_l)
        """
        return np.dot(X, self.W) + self.b

    def gradient(self, x):
        """
        Computes gradients of outputs of linear layer on a single example with respect to the weights and biases
        :param x: input of (size_l_prev,)
        :return: Gradients with respect to weights size (size_l_prev, size_l) and biases size (size_l,)
        """
        return x

    def backward(self, x, g):
        """
        Computes gradients of outputs of linear layer on a single example with respect to the inputs
        :param x: input of (size_l_prev,)
        :param g: gradient delta of (size_l_prev,)
        :return: new gradient back propagated, (size_l,)
        """
        return np.dot(self.W, g)


class ReLU(Layer):
    def __init__(self, name="relu"):
        """
        ReLU layer
        :param name: name of layer
        """
        self._type = "activation"
        self.name = name

    def __str__(self):
        return "{}".format(self.name)

    def forward(self, X):
        """
        Computes forward pass on ReLU activation
        max(x_i, 0)
        :param X: inputs (M, size_l)
        :return: outputs of ReLU (M, size_l)
        """
        return np.maximum(X, 0)

    def backward(self, x, g):
        """
        Computes gradients of outputs of ReLU with respect to inputs
        Gradient is x < 0 and 1 x > 0 (we set to 0 for x = 0)
        :param x: inputs of shape (size_l,)
        :param g: gradient delta of (size_l_prev,)
        :return: new gradient back propagated, (size_l,)
        """
        return np.squeeze(g * np.array(x > 0).astype(float))


class Softmax(Layer):
    def __init__(self, name="softmax"):
        """
        Softmax output layer
        :param name: name of layer
        """
        self._type = "activation"
        self.name = name

    def forward(self,  X):
        """
        Computes forward pass on softmax activation
        :param X: inputs (M, size_l)
        :return: outputs of softmax (M, size_l)
        """
        exps = np.exp(X)
        return exps / np.sum(exps, axis=0, keepdims=True)

    def backward(self, x, g):
        """
        Just pass on the gradient, the actual gradient is computed in
        CrossEntropySoftmax for numerical stability
        :param x: inputs of shape (size_l,)
        :param g: gradient delta of (size_l_prev,)
        :return: new gradient back propagated, (size_l,)
        """
        return g



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


class SGD:
    def __init__(self, model, loss, learning_rate=1e-3):
        """
        Init a SGD optimiser
        :param graph: the graph of the ANN
        :param learning_rate: learning rate for updates
        """
        self.model = model
        self.loss = loss
        self.learning_rate = learning_rate

    def step(self, X, Y):
        """
        Take one step of learning on the graph
        :param X: input batch data
        :param Y: input batch labels
        :return: None
        """
        for x, y in zip(X, Y):
            layer = self.model.inputs
            # print("x", x.shape)
            self._step(layer, x[None], y)

    def _step(self, layer, x, y):
        x_next = np.squeeze(layer.forward(x))

        # base case recursion is the loss
        if layer is self.model.outputs:
            # first evaluate loss
            g = self.loss.gradient(x_next, y)
            # print("error loss", g.shape)
            # print("layer backward", layer.backward(x, g).shape)
            return layer.backward(x, g)

        # pass the gradient on
        g = self._step(layer.next, x_next, y)
        # print ("g", g.shape)

        # update layer if has weights
        if layer._type == "dense":
            # print("g after relu", g.shape)
            # print(layer.gradient(x).shape)
            # print("W", layer.W.shape)
            # print(np.outer(g, layer.gradient(x)).shape)
            layer.W -= self.learning_rate * np.outer(layer.gradient(x), g)
            layer.b -= self.learning_rate * g

        # forward pass and backward passes recursively
        if hasattr(layer, "backward"):
            return layer.backward(x, g)


class Model:
    def __init__(self, inputs, outputs):
        """
        Model defines the structure of computation
        :param inputs: input layer
        :param outputs: output layer
        """
        self.inputs = inputs
        self.outputs = outputs

    def __str__(self):
        o = []
        node = self.inputs
        o.append(str(node))
        while node is not None:
            if not hasattr(node, "next"):
                break
            node = node.next
            o.append(str(node))

        return "\n".join(o)

    def predict(self, X):
        """
        Computes a forward pass on the data batch.
        :param X: input batch
        :return: the outputs of model
        """
        node = self.inputs
        while node is not None:
            if not hasattr(node, "next"):
                break
            node = node.next
            X = node.forward(X)
        return X


def run_XOR_regression():
    """
    Run a simple XOR example regression
    """
    inputs = Inputs(2)
    x = Dense(2, 32)(inputs)
    x = ReLU()(x)
    x = Dense(32, 1)(x)
    m = Model(inputs, x)
    print(m)
    ls = MeanSquaredError(1)
    opt = SGD(m, ls)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    Y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    logits = m.predict(X)
    acc = ls.forward(logits, Y)
    print("logits", logits)
    print("acc", acc)
    ts = [0]
    accs = [acc]
    for i in range(1000):
        ts.append(i+1)
        opt.step(X, Y)
        logits = m.predict(X)
        acc = ls.forward(logits, Y)
        accs.append(acc)

    print("logits", logits)
    print("acc", acc)
    plt.plot(ts, accs)
    plt.xlabel("Test Accuracy")
    plt.ylabel("Number of timesteps")
    plt.show()

def run_XOR_classification():
    """
    Run a simple XOR example classification
    """
    inputs = Inputs(2)
    x = Dense(2, 16)(inputs)
    x = ReLU()(x)
    x = Dense(16, 2)(x)
    x = Softmax()(x)
    m = Model(inputs, x)
    print(m)
    ls = CrossEntropySoftmax(2)
    opt = SGD(m, ls, 1e-2)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    Y = np.array([[1, 0],
                  [0, 1],
                  [0, 1],
                  [1, 0]])
    logits = m.predict(X)
    acc = np.sum(np.argmax(logits, axis=1) == np.argmax(Y, axis=1)) / Y.shape[0]
    print("logits", logits)
    print("acc", acc)
    ts = [0]
    accs = [acc]
    for i in range(50):
        ts.append(i + 1)
        opt.step(X, Y)
        logits = m.predict(X)
        acc = np.sum(np.argmax(logits, axis=1) == np.argmax(Y, axis=1)) / Y.shape[0]
        accs.append(acc)
    logits = m.predict(X)
    print("logits", logits)
    acc = np.sum(np.argmax(logits, axis=1) == np.argmax(Y, axis=1)) / Y.shape[0]
    print("acc", acc)
    plt.plot(ts, accs)
    plt.xlabel("Test Accuracy")
    plt.ylabel("Number of timesteps")
    plt.show()


def run_MNIST():
    """
    Run a small feed forward neural network on subset of MNIST
    """
    # Read in data
    def read_data(path):
        with open(path) as f:
            data = f.readlines()
        all_values = np.asfarray([line.split(',') for line in data])

        X = (all_values[:, 1:] / 255.0 * 0.98) + 0.01  # Scale inputs for sigmoid activation's best (0,1) range
        labels = np.array(all_values[:, 0], dtype=np.int32)  # First number is label
        Y = np.zeros((len(labels), 10))
        Y[range(len(labels)), labels] = 1

        return X, Y

    Xtst, Ytst = read_data('datasets/mnist_train_100.csv')
    Xtrn, Ytrn = read_data('datasets/mnist_test_10.csv')

    inputs = Inputs(784)
    x = Dense(784, 128)(inputs)
    x = ReLU()(x)
    x = Dense(128, 128)(x)
    x = ReLU()(x)
    x = Dense(128, 10)(x)
    x = Softmax()(x)
    m = Model(inputs, x)
    print(m)
    ls = CrossEntropySoftmax(10)
    opt = SGD(m, ls)
    logits = m.predict(Xtst)
    acc = np.sum(np.argmax(logits, axis=1) == np.argmax(Ytst, axis=1)) / Ytst.shape[0]
    print("pred", np.argmax(logits[:3], axis=1))
    print("targ", np.argmax(Ytst[:3], axis=1))
    print("acc", acc)
    ts = [0]
    accs = [acc]
    for i in range(1000):
        ts.append(i + 1)
        opt.step(Xtrn, Ytrn)
        logits = m.predict(Xtst)
        acc = np.sum(np.argmax(logits, axis=1) == np.argmax(Ytst, axis=1)) / Ytst.shape[0]
        accs.append(acc)
    print("pred", np.argmax(logits[:3], axis=1))
    print("targ", np.argmax(Ytst[:3], axis=1))
    acc = np.sum(np.argmax(logits, axis=1) == np.argmax(Ytst, axis=1)) / Ytst.shape[0]
    print("acc", acc)
    plt.plot(ts, accs)
    plt.xlabel("Test Accuracy")
    plt.ylabel("Number of timesteps")
    plt.show()


if __name__ == "__main__":
    run_XOR_regression()
    # run_XOR_classification()
    # run_MNIST()