import numpy as np

from networks.base.function.Function import ReLU, Sigmoid
from networks.base.function.Loss import MSE
from networks.base.init.WeightsInitializerGetter import WeightsInitializerGetter
from networks.base.layer.Layer import Layer
from networks.base.layer.LayerError import EmptyLayerError
from networks.base.optimizer.AdaDelta import AdaDelta
from networks.base.optimizer.Momentum import Momentum
from networks.base.optimizer.Optimizer import Optimizer
from networks.base.optimizer.RMSProp import RMSProp
from networks.base.optimizer.SGD import SGD


class Network:
    def __init__(self,
                 loss=None,
                 optimizer: Optimizer = SGD(alpha=1e-1),
                 weights_initializer: str = 'xavier',
                 max_epochs=1e2,
                 reg=1e-4,
                 tol=1e-3,
                 verbose=True,
                 seed=0):

        self.layers = np.array([], dtype=Layer)
        self.weights_initializer = WeightsInitializerGetter.get(weights_initializer)
        self.optimizer = optimizer
        self.loss_func = loss
        self.max_epochs = int(max_epochs)
        self.reg = reg
        self.tol = tol
        self._neurons_amount = 0
        self.n_layers = 0

        self.verbose = verbose
        self.seed = seed
        np.random.seed(seed)

    def input_layer(self):
        if self.isEmpty():
            EmptyLayerError.raise_error()
        return self.layers[0]

    def output_layer(self):
        if self.isEmpty():
            EmptyLayerError.raise_error()
        return self.layers[-1]

    def add(self, layer):
        if not self.isEmpty():
            self.__connect_layers__(self.output_layer(), layer)

        self.layers = np.append(self.layers, layer)
        self._neurons_amount += layer.n_neurons
        self.n_layers += 1

    def fit(self, X: np.ndarray, y: np.ndarray):
        from tqdm import trange

        n, m = y.shape
        if m != self.output_layer().n_neurons:
            raise IndexError("""
    Shape converting error:
    <Output layer> have {} neurons
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
            loss += self.backward(yi)

        return loss

    def forward(self, x):
        self.input_layer().__set_inputs__(x)
        for layer in self.layers[:-1]:
            layer.activate_and_forward()

        return self.output_layer().activate()

    def backward(self, y):
        layer = self.output_layer()
        loss = self.loss_func.loss(layer.result, y)
        errors = self.loss_func.gradient(layer.result, y)

        while True:
            back_layer = layer.back_layer
            activation_gradient = layer.activation_gradient()
            errors_gradient = (errors * activation_gradient).reshape(1, -1)
            back_error = errors_gradient @ back_layer.neuron_weights.T

            weights_gradient = back_layer.activation_outputs.reshape(-1, 1) @ errors_gradient

            self.optimizer.step(layer=back_layer,
                                neuron_gradient=weights_gradient,
                                biased_gradient=errors_gradient)

            layer = back_layer
            errors = back_error

            if layer.isInput:
                break

        return loss

    def predict(self, X):
        return np.array([self.forward(x) for x in X])

    def isEmpty(self):
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


if __name__ == '__main__':
    optimizer = AdaDelta(gamma=0.5, alpha=1)

    net = Network(max_epochs=1e3, optimizer=optimizer, loss=MSE())

    net.add(Layer(1))

    # net.add(Layer(8, ReLU(alpha=0.4)))
    # net.add(Layer(4, Sigmoid()))
    net.add(Layer(16, ReLU(alpha=0.3)))
    # net.add(Layer(16, Sigmoid()))

    net.add(Layer(1))


    def ff(szzz):
        _X = (np.random.rand(szzz) - 0.5) * 5
        return _X.reshape(szzz, 1), (np.sin(_X) * 3 + 4 + 0.01 * np.random.randn(szzz)).reshape(szzz, 1)


    size = 1000
    X_train, Y_train = ff(size)

    import matplotlib.pyplot as plt

    plt.plot(X_train, Y_train, '.')
    plt.show()

    net.fit(X_train, Y_train)
    predicted = net.predict(X_train).reshape(size, 1)

    plt.plot(X_train, predicted, '.')
    plt.show()
    print(net.loss_func.loss(predicted, Y_train))
