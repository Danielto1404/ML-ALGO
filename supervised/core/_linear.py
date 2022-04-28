import numpy as np

from networks.core.layers.layer import Layer
from networks.net import Network


class LinearRegression:
    def __init__(self, input_shape, output_shape, max_epochs=1e3, tol=1e-3, verbose=True, seed=0):
        self._liner_model = Network(max_epochs=max_epochs, verbose=verbose, seed=seed, tol=tol)
        self._liner_model.add(Layer(n_neurons=input_shape))
        self._liner_model.add(Layer(n_neurons=output_shape))

    def fit(self, X, y):
        self._liner_model.fit(X, y)

    def predict(self, X):
        return self._liner_model.predict(X)


model = LinearRegression(2, 1, max_epochs=10_000)
X = np.array([[1, 2]])
y = np.array([[3]])

model.fit(X, y)
print(model.predict(X))
