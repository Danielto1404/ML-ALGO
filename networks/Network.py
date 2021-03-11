import numpy as np

from networks.base.function.Function import ReLU
from networks.base.function.Loss import MSE
from networks.base.layer.Layer import Layer
from networks.test.test_generator import get_Y, get_X


class Network:
    def __init__(self, loss=None, max_iterations=1e2, reg=1e-4, lr=1e-5, tol=1e-3, seed=0):
        self.layers = np.array([], dtype=Layer)

        self.loss_func = loss
        self.max_iterations = int(max_iterations)
        self.reg = reg
        self.lr = lr
        self.tol = tol
        self._neurons_amount = 0
        self.n_layers = 0

        self.seed = seed
        np.random.seed(seed)

    def input_layer(self):
        return self.layers[0]

    def output_layer(self):
        return self.layers[-1]

    def add(self, layer: Layer):
        if not self.isEmpty():
            self.__connect_layers__(self.output_layer(), layer)

        self.layers = np.append(self.layers, layer)
        self._neurons_amount += layer.n_neurons
        self.n_layers += 1

    def fit(self, X: np.ndarray, y: np.ndarray):
        from tqdm import trange
        n, z = X.shape
        _, m = y.shape
        if m != self.output_layer().n_neurons:
            raise IndexError("""
    Shape converting error:
    <Output layer> have {} neurons
    <Y>            have {} targets
    """.format(self.output_layer().n_neurons, m))

        for _ in trange(self.max_iterations):
            ind = np.random.randint(n)
            self.forward(X[ind])
            loss = self.backward(y[ind])
        # return loss

    def forward(self, x):
        self.input_layer().__set_inputs__(x)
        for i in np.arange(self.n_layers - 1):
            self.layers[i].activate_and_forward()

        return self.output_layer().activate()

    def backward(self, y):
        layer = self.output_layer()
        loss = self.loss_func.loss(layer.result, y)
        errors = self.loss_func.gradient(layer.result, y)
        back_layer = layer.back_layer

        while True:
            activation_gradient = layer.activation_gradient()
            errors_gradient = (errors * activation_gradient).reshape(1, -1)
            back_error = errors_gradient @ back_layer.neuron_weights.T

            weights_gradient = back_layer.activation_outputs.reshape(-1, 1) @ errors_gradient

            back_layer.neuron_weights -= self.lr * weights_gradient
            back_layer.biased_weights -= self.lr * errors_gradient

            layer = back_layer
            errors = back_error

            if layer.isFirst:
                break

        return loss

    def predict(self, X):
        return np.array([self.forward(x) for x in X])

    def isEmpty(self):
        return self.n_layers == 0

    def __init_weights__(self, n_inputs, n_outputs):
        return np.random.uniform(low=-1, high=1, size=(n_inputs, n_outputs)) / np.sqrt(n_inputs)

    def __connect_layers__(self, back_layer: Layer, next_layer: Layer):
        back_layer.next_layer = next_layer
        next_layer.back_layer = back_layer
        back_layer.neuron_weights = self.__init_weights__(back_layer.n_neurons, next_layer.n_neurons)
        back_layer.biased_weights = np.zeros((1, next_layer.n_neurons))

    def __str__(self):
        return "Network with {} non-biased neurons".format(self._neurons_amount)


if __name__ == '__main__':
    net = Network(max_iterations=3e5, loss=MSE(), lr=1e-4, seed=239)

    relu = ReLU(alpha=0)

    input_layer = Layer(n_neurons=3)
    mid = Layer(n_neurons=10, activation=relu)
    output_layer = Layer(n_neurons=1, activation=relu)

    net.add(input_layer)
    net.add(mid)
    net.add(output_layer)

    net.fit(get_X(), get_Y())
    print(net.predict([[150, 150, 150]]))
