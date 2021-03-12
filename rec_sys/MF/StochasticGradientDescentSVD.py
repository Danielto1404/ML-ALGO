import scipy.sparse as sp
from tqdm import trange

from rec_sys.MF.CoreMF import CoreMF


class StochasticGradientDescentSVD(CoreMF):
    def predict(self, i, j):
        return super().predict(i, j) + self.user_bias[i] + self.item_bias[j] + self.bias

    def fit(self, user_to_item: sp.csr_matrix):
        self.__fit_preparation__(user_to_item, init_biases=True)

        tqdm_range = trange(self.iterations, desc='Epochs', colour='green')

        for it in tqdm_range:
            i, j = self.__random_nonzero_indices__()
            error = self.predict(i, j) - user_to_item[i, j]

            self.user_bias[i] -= self.learning_rate \
                                 * (error + self.beta * self.user_bias[i])

            self.item_bias[j] -= self.learning_rate \
                                 * (error + self.beta * self.item_bias[j])

            self.user_factors[i] -= self.learning_rate \
                                    * (self.alpha * self.user_factors[i] + error * self.item_factors[j])

            self.item_factors[j] -= self.learning_rate \
                                    * (self.alpha * self.item_factors[j] + error * self.user_factors[i])

            if it % 10_001 == 0 and self.calculate_loss:
                self.rmse(user_to_item, tqdm_range=tqdm_range)
