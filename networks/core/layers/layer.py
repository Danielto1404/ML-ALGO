import numpy as np

from networks.core.functions.activations import ActivationFunction, Id
from networks.core.layers.layer_error import EmptyLayerError


class Layer:
    """
    Represents single layers in neural network.
    Core block of build neural network.

    --- Attributes ---

    n_neurons: int
        default = 1

        amount of neurons in the layers.

    activation: Function (*networks.core.functions.Function*)
        Implemented functions: **ReLU, Sigmoid, Id**.

    dropout: float [0; 1]
        default = 0

        **probability** to turn off **random** neuron of current layers in training mode.

    """

    def __init__(self, n_neurons: int = 1, activation_func: ActivationFunction = Id(), dropout: float = 0):
        """

        :param n_neurons:  amount of neurons.
        :param activation_func: activation functions
            (Use one from 'networks.core.functions.Activation' module)
        :param dropout:    probability to turn off random neuron of current layers in training mode
        """
        self.n_neurons = n_neurons
        self.n_biased = n_neurons + 1
        self.activation_func = activation_func
        self.dropout = dropout

        self.weighted_outputs = None
        self.activation_outputs = None
        self.inputs = None

        self.neuron_weights = None
        self.biased_weights = None
        self.result = None

        self.next_layer = None
        self.back_layer = None

    def __set_inputs__(self, inputs: np.array):
        """
        Sets inputs from given array values.
        Used to forward outputs to the next layers

        :param inputs:
            np.array (1 x n), where n is amount of neurons on current layers.
        """
        self.inputs = inputs

    def activate(self):
        """
        Applies **activation** functions to layers inputs.

        :return: np.array (1 x n_neurons)
        """
        self.result = self.activation_func.activate(self.inputs)
        return self.result

    def activation_gradient(self):
        """
        Returns gradient of **activation** functions
        calculated on **forwarding** block values

        :return: np.array (1 x n_neurons)
        """
        return self.activation_func.gradient(self.inputs)

    def activate_and_forward(self):
        """
        Calculates weighted outputs on **activated** neurons and forwards them to the **next layers** as an **input**.
        """
        if self.next_layer is None:
            raise EmptyLayerError("""
            Impossible to forward outputs, next layers is empty.
            Use 'activate' to get the result of current layers.
            """)
        self.activation_outputs = self.activation_func.activate(self.inputs)
        self.weighted_outputs = self.activation_outputs @ self.neuron_weights + self.biased_weights
        self.next_layer.__set_inputs__(self.weighted_outputs)

    def isInput(self):
        """
        :return: boolean values that indicates whether current layers is an input layers in neural network
        """
        return self.back_layer is None

    def isOutput(self):
        """
        :return: boolean values that indicates whether current layers is an output layers in neural network
        """
        return self.next_layer is None

    def __str__(self):
        return """
    activation: {}
    dropout   : {}
    neurons   : {}
    """.format(self.activation_func, self.dropout, self.n_neurons)
