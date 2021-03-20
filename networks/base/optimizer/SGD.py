from networks.base.layer.Layer import Layer
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

    def __init__(self, alpha=1e-4, l2_reg=0):
        """

        :param alpha: learning rate
        """
        super(SGD, self).__init__()
        self.alpha = alpha
        self.l2_reg = l2_reg

    def step(self, layer: Layer, neuron_gradient, biased_gradient):
        layer.neuron_weights -= self.alpha * (neuron_gradient + self.l2_reg * layer.neuron_weights)
        layer.biased_weights -= self.alpha * biased_gradient

    def __str__(self):
        return "SGD optimizer [ alpha:= {} ]".format(self.alpha)

    def __repr__(self):
        return str(self)
