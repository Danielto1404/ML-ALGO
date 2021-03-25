import numpy as np


class MiniBatch:
    def __init__(self, X: np.array, y: np.array, n, batch_size=1, shuffle=True):
        """
        Creates iterator throw given data

        :param X: features array
        :param y: marks array
        :param n: number of elements
        :param batch_size: mini-batch size
        :param shuffle: check whether data needed to be shuffled
        """
        self.X = X
        self.y = y

        self.n = n
        self.k = 0
        self.batch_size = batch_size
        self.shuffle = shuffle

        if self.shuffle:
            self.X, self.y = self.__shuffle__(X=self.X, y=self.y, n=self.n)

    def __iter__(self):
        return self

    def __next__(self):
        if self.n <= self.batch_size * self.k:
            raise StopIteration

        start = self.k * self.batch_size
        end = start + self.batch_size
        self.k += 1

        return self.X[start:end], self.y[start:end]

    @staticmethod
    def __shuffle__(X, y, n):
        indices = np.arange(n)
        np.random.seed(indices)

        X_, y_ = [], []

        for i in indices:
            X_.append(X[i])
            y_.append(y[i])

        return np.array(X_), np.array(y_)

    def __reset_index__(self):
        self.k = 0

        if self.shuffle:
            self.X, self.y = self.__shuffle__(self.X, self.y, self.n)
