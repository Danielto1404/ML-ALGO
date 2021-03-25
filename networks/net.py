import numpy as np
import matplotlib.pyplot as plt

from networks.core.data.splitter import MiniBatch
from networks.core.functions.activations import ReLU, Sigmoid
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

    def fit(self, X: np.ndarray, y: np.ndarray, batch_size=1, shuffle=True):
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
        data_loader = MiniBatch(X=X, y=y, n=n, batch_size=batch_size, shuffle=shuffle)

        for _ in epochs:
            loss = self.__fit_epoch__(data_loader)
            data_loader.__reset_index__()

            Q = gamma * Q + (1 - gamma) * loss

            if self.verbose:
                epochs.set_postfix_str("{} loss: {}".format(self.loss_func, Q))

    def __fit_epoch__(self, batches: MiniBatch):
        loss = 0
        n = batches.n
        for (_x, _y) in batches:
            self.forward(_x)
            loss += self.backward(_y, n)

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
        return np.array([self.forward(x) for x in X])

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


if __name__ == '__main__':
    def data(size):
        X_t = (np.random.rand(size) - 0.5) * 25
        sigmoid = np.vectorize(lambda x: 1 / (1 + np.e ** -x))
        Y_y = sigmoid(X_t) + X_t ** 2 * np.sin(X_t)
        return X_t.reshape(size, 1), Y_y.reshape(size, 1)


    size = 1000
    X_train, Y_train = data(size)

    optimizer = Adam(gamma=.999, alpha=1e-3, beta=0.97)

    net = Network(max_epochs=1e2, sgd_optimizer=optimizer)
    net.add(Layer(1))
    net.add(Layer(64, ReLU(alpha=0.3)))
    net.add(Layer(64, ReLU(alpha=0.1)))
    net.add(Layer(64, Sigmoid()))
    net.add(Layer(1))

    net.fit(X_train, Y_train, batch_size=1)

    predicted = net.predict(X_train).reshape(size, 1)

    # plt.plot(X_train, predicted, '.', color='red')
    plt.show()
