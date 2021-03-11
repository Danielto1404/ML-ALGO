import numpy as np

from networks.base.function.Function import Id
from networks.base.layer.LayerError import EmptyLayerError


class NeuronWeights:
    """
    Represents matrix of neuron weights to the next layer.
        (n_neurons x n_outputs)
    """

    def __init__(self, n_neurons, n_outputs):
        self.weights = np.random.random((n_neurons, n_outputs))

    def multiply(self, inputs):
        """
        :param inputs: vector of inputs (1 x n_neurons)

        :return: vector of outputs (1 x n_outputs)
        """
        return inputs @ self.weights


class Layer:
    """
    Represents single layer in neural network.
    Core block of build neural network.

    --- Attributes ---

    n_neurons: int
       amount of neurons in the layer.

    activation: Function (*networks.base.function.Function*)
        Implemented functions: **ReLU, Sigmoid, Id**.

    dropout: float [0; 1]
        default = 0

        **probability** to turn off **random** neuron of current layer in training mode.

    """

    def __init__(self, n_neurons: int, activation=Id(), dropout=0):
        """

        :param n_neurons:  amount of neurons.
        :param activation: activation function
            (Use one from 'networks.base.function.Activation' module)
        :param dropout:    probability to turn off random neuron of current layer in training mode
        """
        self.n_neurons = n_neurons
        self.n_biased = n_neurons + 1
        self.activation_func = activation
        self.dropout = dropout

        self.weighted_outputs = None
        self.activation_outputs = None
        self.inputs = None

        self.neuron_weights = None
        self.biased_weights = None

        self.next_layer = None
        self.back_layer = None
        self.result = None

    def __set_inputs__(self, inputs: np.array):
        """
        Sets inputs from given array values.
        Used to forward outputs to the next layer

        :param inputs:
            np.array (1 x n), where n is amount of neurons on current layer.
        """
        self.inputs = inputs

    def isFirst(self):
        """
        Check whether current layer is first in neural network.

        :return: bool
        """
        return self.back_layer is None

    def activate(self):
        """
        Applies **activation** function to layer inputs.

        :return: np.array (1 x n_neurons)
        """
        self.result = self.activation_func.activate(self.inputs)
        return self.result

    def activation_gradient(self):
        """
        Returns gradient of **activation** function
        calculated on **forwarding** block values

        :return: np.array (1 x n_neurons)
        """
        return self.activation_func.gradient(self.inputs)

    def activate_and_forward(self):
        """
        Calculates weighted outputs on **activated** neurons and forwards them to the **next layer** as an **input**
        """
        if self.next_layer is None:
            raise EmptyLayerError("""
            Impossible to forward outputs, next layer is empty.
            Use 'activate' to get the result of current layer.
            """)
        self.activation_outputs = self.activation_func.activate(self.inputs)
        self.weighted_outputs = self.activation_outputs @ self.neuron_weights + self.biased_weights
        self.next_layer.__set_inputs__(self.weighted_outputs)

    def __str__(self):
        return """
    activation: {}
    dropout   : {}
    neurons   : {}
    inputs    : {}
    outputs   : {}
    """.format(self.activation_func, self.dropout, self.n_neurons, self.inputs, self.weighted_outputs)
