from networks.base.layer.Layer import Layer
from networks.base.optimizer.Optimizer import Optimizer


class Momentum(Optimizer):
    def __init__(self, gamma=0, alpha=1e-2):
        super(Momentum, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def step(self, layer: Layer, neuron_gradient, biased_gradient):
        neuron_weights_avg, biased_weights_avg = self[layer]

        neuron_weights_avg = self.gamma * neuron_weights_avg + (1 - self.gamma) * neuron_gradient
        biased_weights_avg = self.gamma * biased_weights_avg + (1 - self.gamma) * biased_gradient

        self[layer] = (neuron_weights_avg, biased_weights_avg)

        layer.neuron_weights -= self.alpha * neuron_weights_avg
        layer.biased_weights -= self.alpha * biased_weights_avg

    def __str__(self):
        return "Momentum optimizer [ gamma:= {} | alpha:= {} ]".format(self.gamma, self.alpha)

    def __repr__(self):
        return str(self)
