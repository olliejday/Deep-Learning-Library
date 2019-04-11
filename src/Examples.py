import numpy as np
import matplotlib.pyplot as plt
import os
from Layers import Inputs, Dense
from Activations import ReLU, Softmax
from Losses import MeanSquaredError, CrossEntropySoftmax
from Core import Model
from Optimisers import SGD

def run_XOR_regression():
    """
    Run a simple XOR example regression
    """
    inputs = Inputs(2)
    x = Dense(32)(inputs)
    x = ReLU()(x)
    x = Dense(1)(x)
    m = Model(inputs, x)
    print(m)
    ls = MeanSquaredError()
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
    mse = ls.forward(logits, Y)
    print("logits", logits)
    print("mse", -mse)
    ts = [0]
    mses = [mse]
    for i in range(1000):
        ts.append(i+1)
        opt.step(X, Y)
        logits = m.predict(X)
        mse = ls.forward(logits, Y)
        mses.append(mse)

    print("logits", logits)
    print("mse", -mse)
    plt.plot(ts, mses)
    plt.xlabel("Test MSE")
    plt.ylabel("Number of timesteps")
    plt.show()

def run_XOR_classification():
    """
    Run a simple XOR example classification
    """
    inputs = Inputs(2)
    x = Dense(16)(inputs)
    x = ReLU()(x)
    x = Dense(2)(x)
    x = Softmax()(x)
    m = Model(inputs, x)
    print(m)
    ls = CrossEntropySoftmax()
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
    curdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    Xtst, Ytst = read_data(curdir + '/datasets/mnist_train_100.csv')
    Xtrn, Ytrn = read_data(curdir + '/datasets/mnist_test_10.csv')

    inputs = Inputs(784)
    x = Dense(128)(inputs)
    x = ReLU()(x)
    x = Dense(128)(x)
    x = ReLU()(x)
    x = Dense(10)(x)
    x = Softmax()(x)
    m = Model(inputs, x)
    print(m)
    ls = CrossEntropySoftmax()
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
    EXAMLPE_TO_RUN = 3
    if EXAMLPE_TO_RUN == 1:
        run_XOR_regression()
    if EXAMLPE_TO_RUN == 2:
        run_XOR_classification()
    if EXAMLPE_TO_RUN == 3:
        run_MNIST()