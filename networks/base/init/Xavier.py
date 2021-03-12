from networks.base.init.WeightsInitializer import WeightsInitializer
import numpy as np


class Xavier(WeightsInitializer):

    def init_for_neurons(self, n_input, n_outputs):
        """
        Returns matrix of (n_inputs x n_outputs) random initialized weights
        from ( -1 / sqrt(n_input); 1 / sqrt(n_input) )

        :param n_input:   number of neurons which multiplies by weights
        :param n_outputs:

        :return:
        """
        sqrt_6 = np.sqrt(6)
        return np.random.uniform(-sqrt_6, sqrt_6, (n_input, n_outputs)) / np.sqrt(n_input + n_outputs)

    def init_for_biases(self, n_input, n_output):
        return np.zeros((1, n_output))
