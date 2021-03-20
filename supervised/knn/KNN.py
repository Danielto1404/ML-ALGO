import numpy as np
from supervised.knn.base.Metric import Metric, Euclidean


class KNN:
    """
    Naive implementation of K-Nearest-Neighbours metric algorithm
    """

    def __init__(self, neighbours=1, metric: Metric = Euclidean()):
        """
        :param neighbours: numbers of k-neighbours to aggregate voting.
        :param metric:     metric which applied to feature vectors.
        """
        self.neighbours = neighbours
        self.metric = metric

        self.X, self.y = None, None

    def train(self, X, y):
        """
        Makes copy of given dataset.

        :param X: features
        :param y: labels
        :return:
        """
        self.X = X.copy_with_weights()
        self.y = y.copy_with_weights()

    def predict(self, X):
        """
        Makes simple voting process using k-nearest-neighbors by specified metric.

        :param X: array of object features.
        :return:  labels for corresponding features
        """
        predicted = []
        for x in X:
            distances = np.fromiter(map(lambda xi: self.metric.distance(x, xi), self.X), dtype=float)
            indices = np.argsort(distances)[:self.neighbours]
            labels = self.y[indices]
            mark = np.argmax(np.bincount(labels))
            predicted.append(mark)

        return predicted
