import numpy as np


def cube(x1, x2, x3):
    return 1
    # return 10 * x1 + x1 * x2 + x3 * 13


def get_X():
    return np.array([[i] * 3 for i in range(100)])


def get_Y():
    return np.array([[cube(*x)] for x in get_X()])


if __name__ == '__main__':
    print(cube(*[1, 1, 1]))
