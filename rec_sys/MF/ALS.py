import scipy.sparse as sp
import numpy as np
from tqdm import tqdm

from rec_sys.MF.CoreMF import CoreMF


class ALS(CoreMF):

    def __init__(self, iterations, factors, alpha, confidence, seed, calculate_loss=True):
        """
        Attributes
        ----------
        confidence: float
          user-item confidence regularization parameter C{u,i} = 1 + confidence * R{u,i}
        """
        super().__init__(iterations=iterations,
                         factors=factors,
                         learning_rate=None,
                         alpha=alpha,
                         beta=None,
                         seed=seed,
                         calculate_loss=calculate_loss)
        self.confidence = confidence

    def fit(self, user_to_item: sp.csr_matrix):
        self.__fit_preparation__(user_to_item)

        implicit_values = user_to_item.toarray()
        n_users, n_items = user_to_item.shape

        # Preference matrix user-to-item
        P = np.where(implicit_values > 0, 1, 0)
        P_t = P.T

        # Confidence matrix user-to-item
        C = 1 + self.confidence * implicit_values
        C_t = C.T

        # Identity regularization matrix
        alpha_identity = self.alpha * np.eye(self.factors)  # factors x factors

        tqdm_range = tqdm(np.arange(self.iterations), desc='Epochs', colour='green')

        def als_step(n, fixed, latent, preference_matrix, confidence_matrix):
            Y = fixed  # m x factors
            YT = fixed.T  # factors x m
            YTY = YT @ Y  # factors x factors

            for j in np.arange(n):
                # X[j] * (YT * C[j] * Y + alpha * E) = YT * C[j] * P[j]
                # Faster way to calculate [ YT * C[j] * Y ]: [ YT * Y + YT * (C[j] - E) * Y ]

                confidence = confidence_matrix[j]  # 1 x m
                preference = preference_matrix[j]  # 1 x m

                nonzero_mask = preference > 0
                YT_Cj = YT[:, nonzero_mask] * confidence[nonzero_mask]
                YT_Cj_Pj = np.sum(YT_Cj, axis=1)

                YT_Cj = YT[:, nonzero_mask] * (confidence - 1)[nonzero_mask]  # factors x nonzero
                YT_Cj_Y = YT_Cj @ Y[nonzero_mask, :]  # factors x factors

                latent[j] = np.linalg.solve(YTY + YT_Cj_Y + alpha_identity, YT_Cj_Pj)

        for _ in tqdm_range:
            # Users learning
            als_step(n_users, self.item_factors, self.user_factors, P, C)

            # Items learning
            als_step(n_items, self.user_factors, self.item_factors, P_t, C_t)

            if self.calculate_loss:
                self.rmse(user_to_item, tqdm_range, n_elements=10_000)
