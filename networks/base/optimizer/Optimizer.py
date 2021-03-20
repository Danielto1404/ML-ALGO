from copy import deepcopy

from networks.base.layer.Layer import Layer


class Optimizer:
    def __init__(self):
        self.__layer_weights__ = {}
        self.is_inited = False

    def __add_layer__(self, layer: Layer):
        neuron_weights = deepcopy(layer.neuron_weights)
        biased_weights = deepcopy(layer.biased_weights)
        self.__layer_weights__[layer] = (neuron_weights, biased_weights)

    def __setitem__(self, key, value):
        if isinstance(key, Layer):
            if not isinstance(value, tuple):
                value = tuple(value)

            self.__layer_weights__[key] = value
        else:
            raise KeyError('Item must be instance of subclass [ networks.base.layer.Layer ]')

    def __getitem__(self, item):
        if isinstance(item, Layer):
            return self.__layer_weights__[item]

        raise KeyError('Item must be instance of subclass [ networks.base.layer.Layer ]')

    def step(self, layer: Layer, neuron_gradient, biased_gradient):
        raise NotImplementedError
