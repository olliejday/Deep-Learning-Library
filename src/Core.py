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
