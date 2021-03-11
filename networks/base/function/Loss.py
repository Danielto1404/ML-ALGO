import numpy as np


class MSE:
    """
    MSE loss:

           m
     0.5 * ∑  ( predicted(i) - actual(i) )^2
          i=0
    """

    def gradient(self, predicted: np.array, actual: np.array):
        return predicted - actual

    def loss(self, predicted, actual):
        return 0.5 * np.sum(np.square(predicted - actual))
