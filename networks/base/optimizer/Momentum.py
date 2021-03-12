from networks.base.optimizer.Optimizer import Optimizer


class Momentum(Optimizer):
    def __init__(self, gamma, alpha):
        self.gamma = gamma
        self.alpha = alpha
        self.average = None

    def step(self, gradient, layer_index=None):
        if self.average is None:
            self.average = gradient

        self.average = self.gamma * self.average + (1 - self.gamma) * gradient
        return self.alpha * self.average

    def update(self, gradient, layer_index):
        pass

    def __str__(self):
        return "Momentum optimizer [ gamma:= {} | alpha:= {} ]".format(self.gamma, self.alpha)

    def __repr__(self):
        return str(self)
