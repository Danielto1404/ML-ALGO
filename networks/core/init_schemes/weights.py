import numpy as np


class WeightsInitializer:
    def init_for_neurons(self, n_input, n_outputs):
        raise NotImplementedError

    def init_for_biases(self, n_input, n_output):
        raise NotImplementedError


class Xavier(WeightsInitializer):

    def init_for_neurons(self, n_input, n_outputs):
        """
        Returns matrix of (n_inputs x n_outputs) random initialized weights
        from ( 6 / sqrt(n_input + n_output); 6 / sqrt(n_input + n_output) )

        :param n_input:   number of neurons which multiplies by weights
        :param n_outputs:

        :return:
        """
        sqrt_6 = np.sqrt(6)
        return np.random.uniform(-sqrt_6, sqrt_6, (n_input, n_outputs)) / np.sqrt(n_input + n_outputs)

    def init_for_biases(self, n_input, n_output):
        return np.zeros((1, n_output))


def get_weight_initializer(name):
    initializers = {
        'xavier': Xavier(),
    }

    initializer = initializers.get(name)
    if initializer is None:
        raise UnknownWeightsInitializer(unknown_name=name,
                                        possible_names=list(initializers.keys()))

    return initializer


class UnknownWeightsInitializer(Exception):
    def __init__(self, unknown_name, possible_names):
        super(UnknownWeightsInitializer, self).__init__(
            """
                Unknown  weights initializer: {}
                Possible weights initializer: {}
            """.format(unknown_name, possible_names)
        )
