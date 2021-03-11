import numpy as np


class Function:
    def __init__(self):
        self._activate = None
        self._gradient = None

    def activate(self, values):
        return self._activate(values)

    def gradient(self, values):
        return self._gradient(values)


class Id(Function):
    """
    y(x) = x
    """

    def activate(self, values):
        return values

    def gradient(self, values):
        return np.ones(len(values))


class ReLU(Function):
    def __init__(self, alpha=0):
        super().__init__()
        self.alpha = alpha
        self._activate = np.vectorize(lambda x: alpha * x if x <= 0 else x)
        self._gradient = np.vectorize(lambda x: alpha if x <= 0 else 1)

    def __str__(self):
        return """ReLU  [ alpha={} ]""".format(self.alpha)


class Sigmoid(Function):
    def __init__(self):
        super().__init__()

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        self._activate = np.vectorize(sigmoid)
        self._gradient = np.vectorize(lambda x: sigmoid(x) * (1 - sigmoid(x)))

    def __str__(self):
        return "Sigmoid"
