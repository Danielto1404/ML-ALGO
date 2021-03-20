import numpy as np

from networks.base.layer.Layer import Layer
from networks.base.optimizer.Optimizer import Optimizer


class AdaDelta(Optimizer):
    def __init__(self, gamma=0, alpha=1, eps=1e-10):
        super(AdaDelta, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def step(self, layer: Layer, neuron_gradient, biased_gradient):
        # (
        #   (Grad, Delta, Sigma) <- neurones
        #   (Grad, Delta, Sigma) <- biases
        # )

        neuron_squared = neuron_gradient * neuron_gradient
        biased_squared = biased_gradient * biased_gradient

        if not self.is_inited:
            grad_n = neuron_squared
            grad_b = biased_squared

            sigma_n = neuron_gradient
            sigma_b = biased_gradient

            delta_n = sigma_n * sigma_n
            delta_b = sigma_b * sigma_b

            self[layer] = ((grad_n, delta_n, sigma_n), (grad_b, delta_b, sigma_b))
            self.is_inited = True

        neuron_args, bias_args = self[layer]
        grad_n, delta_n, sigma_n = neuron_args
        grad_b, delta_b, sigma_b = bias_args

        grad_n = self.gamma + (1 - self.gamma) * neuron_squared
        grad_b = self.gamma + (1 - self.gamma) * biased_squared

        sigma_n = neuron_gradient * (np.sqrt(delta_n) + self.eps) / (np.sqrt(grad_n) + self.eps)
        sigma_b = biased_gradient * (np.sqrt(delta_b) + self.eps) / (np.sqrt(grad_b) + self.eps)

        delta_n = self.gamma * delta_n + (1 - self.gamma) * sigma_n * sigma_n
        delta_b = self.gamma * delta_b + (1 - self.gamma) * sigma_b * sigma_b

        self[layer] = ((grad_n, delta_n, sigma_n), (grad_b, delta_b, sigma_b))

        layer.neuron_weights -= self.alpha * sigma_n
        layer.biased_weights -= self.alpha * sigma_b