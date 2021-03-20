import numpy as np

from networks.base.layer.Layer import Layer
from networks.base.optimizer.Optimizer import Optimizer


class RMSProp(Optimizer):
    def __init__(self, gamma=0, alpha=1e-2, eps=1e-10):
        super(RMSProp, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def step(self, layer: Layer, neuron_gradient, biased_gradient):
        neuron_squared = neuron_gradient * neuron_gradient
        biased_squared = biased_gradient * biased_gradient

        if not self.is_inited:
            self[layer] = (neuron_squared, biased_squared)
            self.is_inited = True

        neuron_weights_avg, biased_weights_avg = self[layer]

        neuron_weights_avg = self.gamma * neuron_weights_avg + (1 - self.gamma) * neuron_squared
        biased_weights_avg = self.gamma * biased_weights_avg + (1 - self.gamma) * biased_squared

        self[layer] = (neuron_weights_avg, biased_weights_avg)

        layer.neuron_weights -= self.alpha * neuron_gradient / (np.sqrt(neuron_weights_avg) + self.eps)
        layer.biased_weights -= self.alpha * biased_gradient / (np.sqrt(biased_weights_avg) + self.eps)
