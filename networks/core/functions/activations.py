import numpy as np


class ActivationFunction:
    """
    **Superclass** of all activation functions.

    If you need **custom activation** function you can subclass from this class and implement methods:
        - **def activate(self, values)**
        - **def gradient(self, values)**

    """

    def __init__(self):
        self._activate = None
        self._gradient = None

    def activate(self, values):
        return self._activate(values)

    def gradient(self, values):
        return self._gradient(values)

    def __repr__(self):
        return str(self)


class Id(ActivationFunction):
    """
    y(x) = x
    """

    def activate(self, values):
        return values

    def gradient(self, values):
        return np.ones(len(values))

    def __str__(self):
        return "Id"


class ReLU(ActivationFunction):
    """
    **{ y(x) = x, x > 0  |  y(x) = alpha * x, x <= 0 }**
    """

    def __init__(self, alpha=0):
        super(ReLU, self).__init__()
        self.alpha = alpha
        self._activate = np.vectorize(lambda x: alpha * x if x <= 0 else x)
        self._gradient = np.vectorize(lambda x: alpha if x <= 0 else 1)

    def __str__(self):
        return "ReLU  [ alpha={} ]".format(self.alpha)


class LeakyReLU(ReLU):
    pass


class Sigmoid(ActivationFunction):
    def __init__(self):
        super(Sigmoid, self).__init__()

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def grad(x):
            sig = sigmoid(x)
            return sig * (1 - sig)

        self._activate = np.vectorize(sigmoid)
        self._gradient = np.vectorize(grad)

    def __str__(self):
        return "Sigmoid"


class SoftMax(ActivationFunction):
    def __init__(self):
        super(SoftMax, self).__init__()

    def activate(self, values):
        exps = np.exp(values)
        return exps / np.sum(exps)

    def gradient(self, values):
        pass

    def __str__(self):
        return "SoftMax"
