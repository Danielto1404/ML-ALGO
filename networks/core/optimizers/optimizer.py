from copy import deepcopy

import numpy as np

from networks.core.layers.layer import Layer


class Optimizer:
    def __init__(self):
        self.__layer_weights__ = {}
        self.__is_inited__ = {}

    def __add_layer__(self, layer: Layer):
        neuron_weights = deepcopy(layer.neuron_weights)
        biased_weights = deepcopy(layer.biased_weights)

        self.__layer_weights__[layer] = (neuron_weights, biased_weights)
        self.__is_inited__[layer] = False

    def isInited(self, layer):
        if isinstance(layer, Layer):
            return self.__is_inited__[layer]

        raise KeyError('Item must be instance of subclass [ networks.core.layers.Layer ]')

    def __mark_as_inited__(self, layer):
        self.__is_inited__[layer] = True

    def __setitem__(self, key, value):
        if isinstance(key, Layer):
            if not isinstance(value, tuple):
                value = tuple(value)

            self.__layer_weights__[key] = value
        else:
            raise KeyError('Item must be instance of subclass [ networks.core.layers.Layer ]')

    def __getitem__(self, item):
        if isinstance(item, Layer):
            return self.__layer_weights__.get(item)

        raise KeyError('Item must be instance of subclass [ networks.core.layers.Layer ]')

    def step(self, layer: Layer, neuron_gradient, biased_gradient):
        raise NotImplementedError


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
        return "SGD optimizers [ alpha:= {} ]".format(self.alpha)

    def __repr__(self):
        return str(self)


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
        return "Momentum optimizers [ gamma:= {} | alpha:= {} ]".format(self.gamma, self.alpha)

    def __repr__(self):
        return str(self)


class AdaGrad(Optimizer):
    def __init__(self, alpha, eps=1e-10):
        super(AdaGrad, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def step(self, layer: Layer, neuron_gradient, biased_gradient):
        neuron_squared = neuron_gradient ** 2
        biased_squared = biased_gradient ** 2

        if not self.isInited(layer):
            self[layer] = (neuron_squared, biased_squared)
            self.__mark_as_inited__(layer)

        neuron_weights_avg, biased_weights_avg = self[layer]

        layer.neuron_weights -= self.alpha / (np.sqrt(neuron_weights_avg + self.eps))
        layer.biased_weights -= self.alpha / (np.sqrt(biased_weights_avg + self.eps))

        self[layer] = (neuron_weights_avg, biased_weights_avg)

        neuron_weights_avg += neuron_squared
        biased_weights_avg += biased_squared


class AdaDelta(Optimizer):
    def __init__(self, gamma=0, alpha=1, eps=1e-10):
        super(AdaDelta, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def step(self, layer: Layer, neuron_gradient, biased_gradient):
        # (
        #   (avg_grad, delta, sigma) <- neurones
        #   (avg_grad, delta, sigma) <- biases
        # )

        neuron_squared = neuron_gradient ** 2
        biased_squared = biased_gradient ** 2

        if not self.isInited(layer):
            avg_grad_n = neuron_squared
            avg_grad_b = biased_squared

            sigma_n = neuron_gradient
            sigma_b = biased_gradient

            delta_n = sigma_n * sigma_n
            delta_b = sigma_b * sigma_b

            self[layer] = ((avg_grad_n, delta_n, sigma_n), (avg_grad_b, delta_b, sigma_b))
            self.__mark_as_inited__(layer)

        ((avg_grad_n, delta_n, sigma_n), (avg_grad_b, delta_b, sigma_b)) = self[layer]

        avg_grad_n = self.gamma * avg_grad_n + (1 - self.gamma) * neuron_squared
        avg_grad_b = self.gamma * avg_grad_b + (1 - self.gamma) * biased_squared

        sigma_n = neuron_gradient * (np.sqrt(delta_n) + self.eps) / (np.sqrt(avg_grad_n) + self.eps)
        sigma_b = biased_gradient * (np.sqrt(delta_b) + self.eps) / (np.sqrt(avg_grad_b) + self.eps)

        delta_n = self.gamma * delta_n + (1 - self.gamma) * sigma_n * sigma_n
        delta_b = self.gamma * delta_b + (1 - self.gamma) * sigma_b * sigma_b

        self[layer] = ((avg_grad_n, delta_n, sigma_n), (avg_grad_b, delta_b, sigma_b))

        layer.neuron_weights -= self.alpha * sigma_n
        layer.biased_weights -= self.alpha * sigma_b


class RMSProp(Optimizer):
    def __init__(self, gamma=0.99, alpha=1e-2, eps=1e-10):
        super(RMSProp, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def step(self, layer: Layer, neuron_gradient, biased_gradient):
        neuron_squared = neuron_gradient ** 2
        biased_squared = biased_gradient ** 2

        if not self.isInited(layer):
            self[layer] = (neuron_squared, biased_squared)
            self.__mark_as_inited__(layer)

        neuron_weights_avg, biased_weights_avg = self[layer]

        neuron_weights_avg = self.gamma * neuron_weights_avg + (1 - self.gamma) * neuron_squared
        biased_weights_avg = self.gamma * biased_weights_avg + (1 - self.gamma) * biased_squared

        self[layer] = (neuron_weights_avg, biased_weights_avg)

        layer.neuron_weights -= self.alpha * neuron_gradient / (np.sqrt(neuron_weights_avg + self.eps))
        layer.biased_weights -= self.alpha * biased_gradient / (np.sqrt(biased_weights_avg + self.eps))


class Adam(Optimizer):
    def __init__(self, gamma, alpha, beta, eps=1e-10):
        super(Adam, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def step(self, layer: Layer, neuron_gradient, biased_gradient):
        # (
        #   (grad, avg_grad) <- neurones
        #   (grad, avg_grad) <- biases
        # )

        neuron_squared = neuron_gradient ** 2
        biased_squared = biased_gradient ** 2

        if not self.isInited(layer):
            grad_n = neuron_squared
            grad_b = biased_squared

            avg_n = neuron_gradient
            avg_b = biased_gradient

            self[layer] = ((grad_n, avg_n), (grad_b, avg_b))
            self.__mark_as_inited__(layer)

        ((grad_n, avg_n), (grad_b, avg_b)) = self[layer]

        grad_n = self.gamma * grad_n + (1 - self.gamma) * neuron_squared
        grad_b = self.gamma * grad_b + (1 - self.gamma) * biased_squared

        avg_n = self.beta * avg_n + (1 - self.beta) * neuron_gradient
        avg_b = self.beta * avg_b + (1 - self.beta) * biased_gradient

        self[layer] = ((grad_n, avg_n), (grad_b, avg_b))

        layer.neuron_weights -= self.alpha * avg_n / (np.sqrt(grad_n + self.eps))
        layer.biased_weights -= self.alpha * avg_b / (np.sqrt(grad_b + self.eps))
