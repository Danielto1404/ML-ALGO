from networks.base.optimizer.Optimizer import Optimizer


class SGD(Optimizer):
    """
    Basic **Stochastic** Gradient Descent with fixed **learning rate**.

    --- Attributes ---

    alpha: float
        learning rate

    l2_reg: float
        **L2** regularization parameter
    """

    def __init__(self, alpha=1e-4):
        """

        :param alpha: learning rate
        """
        self.alpha = alpha

    def step(self, gradient, layer_index=None):
        """

        :param layer_index:
        :param gradient: Loss function gradient

        :return: anti gradient step in increasing function way.
        """
        return self.alpha * gradient

    def update(self, gradient, layer_index):
        pass

    def __str__(self):
        return "SGD optimizer [ alpha:= {} ]".format(self.alpha)

    def __repr__(self):
        return str(self)
