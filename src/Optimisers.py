import numpy as np


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
