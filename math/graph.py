import numpy as np


class Operation:
    def apply(self):
        raise NotImplementedError

    def gradient(self):
        raise NotImplementedError


class Add(Operation):
    def __init__(self, variables: [Operation]):
        self.variables = variables
        self.forward = None
        self.result = None

    def apply(self):
        self.forward = [a.apply() for a in self.variables]
        self.result = np.sum(self.forward)

    def gradient(self):
        pass


class Variable:
    def __init__(self, value):
        self.grad = 0
        self.value = value


class Variable(Operation):
    def __init__(self, value):
        self.value = value

    def apply(self):
        return self.value

    def gradient(self):
        return 1


class Tensor(Operation):
    def apply(self):
        pass

        # return self.tensor

    def gradient(self):
        return np.one
