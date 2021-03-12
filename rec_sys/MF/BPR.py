import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from rec_sys.MF.CoreMF import CoreMF


class BPR(CoreMF):
    def __init__(self, iterations, factors, learning_rate, alpha, seed):
        super().__init__(iterations, factors, learning_rate, alpha, seed=seed, beta=None, calculate_loss=None)
        self.positives = {}
        self.negatives = {}

    def negative_choice(self, u):
        return np.random.choice(self.negatives[u])

    def fit(self, user_to_item: sp.csr_matrix):
        self.__fit_preparation__(user_to_item)

        implicit_values = user_to_item.toarray()
        n_users, n_items = user_to_item.shape

        items_range = np.arange(n_items)
        users_range = np.unique(self.user_indices)

        for u in np.arange(n_users):
            values = implicit_values[u]
            self.positives[u] = items_range[values > 0]
            self.negatives[u] = items_range[values == 0]

        def anti_gradient_step(m, gradient, latent):
            exp = np.exp(-m)
            return self.learning_rate * ((exp / (1 + exp)) * gradient - self.alpha * latent)

        for it in np.arange(self.iterations):
            for u in tqdm(users_range, desc='Epoch {}'.format(it + 1), colour='green'):
                for positive in self.positives[u]:
                    negative = self.negative_choice(u)

                    positive_item = self.item_factors[positive]
                    negative_item = self.item_factors[negative]
                    user_factors = self.user_factors[u]
                    delta = positive_item - negative_item

                    margin = user_factors @ delta.T

                    self.user_factors[u] += anti_gradient_step(margin, delta, user_factors)
                    self.item_factors[positive] += anti_gradient_step(margin, user_factors, positive_item)
                    self.item_factors[negative] += anti_gradient_step(margin, -user_factors, negative_item)
