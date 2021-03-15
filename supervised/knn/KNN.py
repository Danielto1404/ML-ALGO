import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import trange

from supervised.knn.base.Metric import Metric, Euclidean


class KNN:
    """
    Naive implementation of K-Nearest-Neighbours metric algorithm
    """

    def __init__(self, neighbours=1, metric: Metric = Euclidean()):
        """
        :param neighbours: numbers of k-neighbours to aggregate voting.
        :param metric:     metric which applied to feature vectors
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
        self.X = X.copy()
        self.y = y.copy()

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


if __name__ == '__main__':

    X_train, y_train = make_moons(n_samples=500, shuffle=True, noise=1, random_state=239)

    scores = {}

    colours = {
        0: 'red',
        1: 'blue'
    }

    for (x, y) in zip(X_train, y_train):
        plt.plot(x[0], x[1], marker='.', color=colours[y])

    plt.xlabel('$x2$')
    plt.ylabel('$x1$')
    plt.show()

    for k in trange(1, 40, 4):
        knn = KNN(neighbours=k)
        knn.train(X_train, y_train)
        score = accuracy_score(knn.predict(X_train), y_train)
        scores[k] = score

    plt.plot(scores.keys(), scores.values())
    plt.xlabel('K neighbours')
    plt.ylabel('Precision score')
    plt.show()
