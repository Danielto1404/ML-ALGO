import numpy as np

from networks.core.functions.losses import MSE
from networks.core.init_schemes.weights import get_weight_initializer
from networks.core.layers.layer import Layer
from networks.core.layers.layer_error import EmptyLayerError
from networks.core.optimizers.optimizer import Optimizer, Adam


class Network:
    """
    Represents fully connected neural networks.
    (**Multi-layers-Perceptron**)
    """

    def __init__(self,
                 loss=MSE(),
                 sgd_optimizer: Optimizer = Adam(gamma=0.999, alpha=1e-3, beta=0.9),
                 weights_initializer: str = 'xavier',
                 max_epochs=1e2,
                 tol=1e-3,
                 verbose=True,
                 seed=0):

        self.layers = np.array([], dtype=Layer)
        self.weights_initializer = get_weight_initializer(name=weights_initializer)
        self.optimizer = sgd_optimizer
        self.loss_func = loss
        self.max_epochs = int(max_epochs)
        self.tol = tol
        self._neurons_amount = 0
        self.n_layers = 0

        self.verbose = verbose
        self.seed = seed
        np.random.seed(seed)

    def input_layer(self):
        """
        :return: first layers in neural network
        """
        if self.isEmpty():
            raise EmptyLayerError()
        return self.layers[0]

    def output_layer(self):
        """
        :return: last layers in neural network
        """
        if self.isEmpty():
            raise EmptyLayerError()
        return self.layers[-1]

    def add(self, layer):
        """
        Adds fully connected layers to neural network.
        """
        if not self.isEmpty():
            self.__connect_layers__(self.output_layer(), layer)

        self.layers = np.append(self.layers, layer)
        self._neurons_amount += layer.n_neurons
        self.n_layers += 1

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        :param X: features dataset
        :param y: targets dataset
        """
        from tqdm import trange

        n, m = y.shape
        if m != self.output_layer().n_neurons:
            raise IndexError("""
    Shape converting error:
    <Output layers> have {} neurons
    <Y>            have {} targets
    """.format(self.output_layer().n_neurons, m))

        if self.verbose:
            epochs = trange(self.max_epochs, colour='green', desc='Epochs')
        else:
            epochs = np.arange(self.max_epochs)

        Q = 0
        gamma = 0
        for _ in epochs:
            loss = self.__fit_epoch__(X, y, n)
            Q = gamma * Q + (1 - gamma) * loss

            if self.verbose:
                epochs.set_postfix_str("{} loss: {}".format(self.loss_func, Q))

    def __fit_epoch__(self, X, y, n):
        loss = 0
        for i in np.arange(n):
            xi, yi = X[i], y[i]
            _ = self.forward(xi)
            loss += self.backward(yi, n)

        return loss

    def forward(self, x) -> np.ndarray:
        """
        Make a forward pass through neural network with given batch of elements.

        :param x: batch of xs to calculate
        :return:  vector of output for each element in batch
        """
        self.input_layer().__set_inputs__(x)
        for layer in self.layers[:-1]:
            layer.activate_and_forward()

        return self.output_layer().activate()

    def backward(self, y, n) -> np.float64:
        """
        Distributes gradient via backpropagation algorithm and return loss.

        :param y: actual value
        :param n: number of train elements
        :return:  loss functions value on yi with pre-forwarded xi
        """
        layer = self.output_layer()
        loss = self.loss_func.loss(layer.result, y, n)

        errors_gradient = (self.loss_func.gradient(layer.result, y, n)
                           @ layer.activation_gradient()).reshape(1, -1)

        for layer in reversed(self.layers[1:-1]):
            self.optimizer.step(
                layer=layer,
                neuron_gradient=layer.activation_outputs.reshape(-1, 1) @ errors_gradient,
                biased_gradient=errors_gradient
            )

            errors_gradient = errors_gradient @ layer.neuron_weights.T * layer.activation_gradient()

        return loss

    def predict(self, X):
        """

        :param X: Batch of features
        :return:  Predicated output for each element in batch
        """
        return [self.forward(x) for x in X]

    def isEmpty(self):
        """
        :return: True if neural network have 0 layers, False otherwise
        """
        return self.n_layers == 0

    def __connect_layers__(self, back_layer: Layer, next_layer: Layer):
        back_layer.next_layer = next_layer
        next_layer.back_layer = back_layer

        back_layer.neuron_weights = self.weights_initializer.init_for_neurons(n_input=back_layer.n_neurons,
                                                                              n_outputs=next_layer.n_neurons)

        back_layer.biased_weights = self.weights_initializer.init_for_biases(n_input=back_layer.n_neurons,
                                                                             n_output=next_layer.n_neurons)

        self.optimizer.__add_layer__(back_layer)

    def __str__(self):
        return "Network with {} non-biased neurons".format(self._neurons_amount)

    def __repr__(self):
        return str(self)
