import numpy as np

from networks.base.layer.Layer import Layer
from networks.base.optimizer.Optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, gamma, alpha, beta, eps=1e-10):
        super(Adam, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def step(self, layer: Layer, neuron_gradient, biased_gradient):
        # (
        #   (Grad, Avg) <- neurones
        #   (Grad, Avg) <- biases
        # )

        neuron_squared = neuron_gradient * neuron_gradient
        biased_squared = biased_gradient * biased_gradient

        if not self.is_inited:
            grad_n = neuron_squared
            grad_b = biased_squared

            avg_n = neuron_gradient
            avg_b = biased_gradient

            self[layer] = ((grad_n, avg_n), (grad_b, avg_b))
            self.is_inited = True

        ((grad_n, avg_n), (grad_b, avg_b)) = self[layer]

        grad_n = self.gamma * grad_n + (1 - self.gamma) * neuron_squared
        grad_b = self.gamma * grad_b + (1 - self.gamma) * biased_squared

        avg_n = self.beta * avg_n + (1 - self.beta) * neuron_gradient
        avg_b = self.beta * avg_b + (1 - self.beta) * biased_gradient

        self[layer] = ((grad_n, avg_n), (grad_b, avg_b))

        layer.neuron_weights -= self.alpha * avg_n / (np.sqrt(grad_n) + self.eps)
        layer.biased_weights -= self.alpha * avg_b / (np.sqrt(grad_b) + self.eps)
