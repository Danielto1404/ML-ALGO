import numpy as np


class Metric:
    def distance(self, x: np.array, y: np.array):
        raise NotImplementedError


class Minkowski(Metric):
    def __init__(self, dimension):
        self.dimension = dimension

    def distance(self, x: np.array, y: np.array):
        return np.linalg.norm(x - y, ord=self.dimension)


class L1(Minkowski):
    def __init__(self):
        super(L1, self).__init__(dimension=1)


class Euclidean(Minkowski):
    def __init__(self):
        super(Euclidean, self).__init__(dimension=2)


class Cube(Minkowski):
    def __init__(self):
        super(Cube, self).__init__(dimension=3)


class Cosine(Metric):
    def distance(self, x: np.array, y: np.array):
        return np.dot(x, y)
