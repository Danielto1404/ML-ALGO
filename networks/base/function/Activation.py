import numpy as np


class ActivationFunction:
    def __init__(self):
        self._activate = None
        self._gradient = None

    def apply(self, values):
        self._activate(values)

    def gradient(self, values):
        return self._gradient(values)

    def __repr__(self):
        return str(self)


class Id(ActivationFunction):
    """
    y(x) = x
    """

    def apply(self, values):
        return values

    def gradient(self, values):
        return np.ones(len(values))

    def __str__(self):
        return "Id"


class ReLU(ActivationFunction):
    def __init__(self, alpha=0):
        super().__init__()
        self.alpha = alpha
        self._activate = np.vectorize(lambda x: alpha * x if x <= 0 else x)
        self._gradient = np.vectorize(lambda x: alpha if x <= 0 else 1)

    def __str__(self):
        return "ReLU  [ alpha={} ]".format(self.alpha)


class Sigmoid(ActivationFunction):
    def __init__(self):
        super().__init__()

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def grad(x):
            sig = sigmoid(x)
            return sig * (1 - sig)

        self._activate = np.vectorize(sigmoid)
        self._gradient = np.vectorize(grad)

    def __str__(self):
        return "Sigmoid"
