import numpy as np

from networks.base.function.Function import ReLU
from networks.base.function.Loss import MSE
from networks.base.init.WeightsInitializerGetter import WeightsInitializerGetter
from networks.base.layer.Layer import Layer
from networks.base.layer.LayerError import EmptyLayerError
from networks.base.optimizer.Optimizer import Optimizer
from networks.base.optimizer.SGD import SGD


class Network:
    def __init__(self,
                 loss=None,
                 optimizer: Optimizer = SGD(alpha=1e-4),
                 weights_initializer: str = 'xavier',
                 max_iterations=1e2,
                 reg=1e-4,
                 tol=1e-3,
                 seed=0):

        self.layers = np.array([], dtype=Layer)
        self.weights_initializer = WeightsInitializerGetter.get(weights_initializer)
        self.optimizer = optimizer
        self.loss_func = loss
        self.max_iterations = int(max_iterations)
        self.reg = reg
        self.tol = tol
        self._neurons_amount = 0
        self.n_layers = 0

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
        iterations = trange(self.max_iterations, colour='green')
        loss = 0
        for _ in iterations:
            iterations.set_postfix_str("{} loss: {}".format(self.loss_func, loss))
            ind = np.random.randint(n)
            self.forward(X[ind])
            loss = self.backward(y[ind])
            # loss = (1 - self.alpha) + self.alpha * self.backward(y[ind])

    def forward(self, x):
        self.input_layer().__set_inputs__(x)
        for i in np.arange(self.n_layers - 1):
            self.layers[i].activate_and_forward()

        return self.output_layer().activate()

    def backward(self, y):
        layer = self.output_layer()
        loss = self.loss_func.loss(layer.result, y)
        errors = self.loss_func.gradient(layer.result, y)

        for layer in reversed(self.layers[1:-1]):
            errors_gradient = layer.backward(errors)
            errors = errors_gradient @ layer.back_layer.neuron_weight.T

            layer.back_layer.activation_outputs.reshape(-1, 1) @ errors_gradient
            layer.back_layer.neuron_weights -= self.optimizer.step(weights_gradient, 0)
            back_layer.biased_weights -= self.optimizer.step(errors_gradient, 0)


        while True:
            activation_gradient = layer.activation_gradient()
            errors_gradient = (errors * activation_gradient).reshape(1, -1)
            back_error = errors_gradient @ back_layer.neuron_weights.T

            weights_gradient = back_layer.activation_outputs.reshape(-1, 1) @ errors_gradient

            back_layer.neuron_weights -= self.optimizer.step(weights_gradient, 0)
            back_layer.biased_weights -= self.optimizer.step(errors_gradient, 0)

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

        back_layer.neuron_weights = self.weights_initializer.init_for_neurons(back_layer.n_neurons,
                                                                              next_layer.n_neurons)
        back_layer.biased_weights = self.weights_initializer.init_for_biases(back_layer.n_neurons,
                                                                             next_layer.n_neurons)

    def __str__(self):
        return "Network with {} non-biased neurons".format(self._neurons_amount)


if __name__ == '__main__':
    def f(x1, x2, x3):
        return x1 ** 2 + 10 * x2 + 5 * x3


    net = Network(max_iterations=4e5, loss=MSE())
    net.add(Layer(3))
    net.add(Layer(30, ReLU()))
    net.add(Layer(30, ReLU()))
    # net.add(Layer(30, ReLU()))
    net.add(Layer(1))

    # %%

    xs = np.random.rand(1000, 3) * 10
    ys = np.array([f(*a) for a in xs]).reshape(1000, 1)

    net.fit(xs, ys)

    # %%

    x = [0, 0, -1]
    print(f(*x))
    # net.predict([x])
