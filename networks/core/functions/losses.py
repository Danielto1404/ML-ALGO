import numpy as np


class LossFunction:
    def loss(self, predicted, actual, n=1):
        raise NotImplementedError

    def gradient(self, predicted, actual, n=1):
        raise NotImplementedError

    def __repr__(self):
        return str(self)


class MSE(LossFunction):
    """
    MSE loss:

     0.5 * ∑  ( predicted(i) - actual(i) )^2
          i=0
    """

    def loss(self, predicted, actual, n=1):
        return 0.5 / n * np.sum(np.square(predicted - actual))

    def gradient(self, predicted: np.array, actual: np.array, n=1):
        return predicted - actual

    def __str__(self):
        return "MSE"


class CrossEntropy(LossFunction):

    def loss(self, predicted, actual, n=1):
        pass

    def gradient(self, predicted, actual, n=1):
        pass

    def __str__(self):
        return "Cross Entropy"
