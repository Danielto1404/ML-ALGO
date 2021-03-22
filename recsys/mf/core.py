import numpy as np
import scipy.sparse as sp


class CoreMF:
    """
    iterations: int
      Amount of algorithm iterations\n
      default = 1_000_000

    factors: int
      Dimension of latent space\n
      default = 64

    learning_rate: float
      Used only in gradient descent bases algorithms\n
      default = 1e-2

    alpha: float
      (L2 - Ridge regularization)\n
      Regularization value for user-factors / item-factors matrices weights\n
      default = 1e-2

    beta: float
      (L1 - Lasso regularization)\n
      Regularization value for user / item biases\n
      default = 1e-2

    X: 2-D np.ndarray
      User latent matrix (|Users| x factors)

    Y: 2-D np.ndarray
      Item latent matrix (|Items| x factors)
    """

    def __init__(self,
                 iterations: float = 1e6,
                 factors: int = 64,
                 learning_rate: float = 1e-2,
                 alpha: float = 1e-2,
                 beta: float = 1e-2,
                 seed: int = 0,
                 calculate_loss: bool = True):
        self.iterations = int(iterations)
        self.factors = factors
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.calculate_loss = calculate_loss

        # Will be computed in fit functions
        self.user_to_item = None
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.bias = None
        self.nonzero = None
        self.nonzero_count = None

    def __random_nonzero_indices__(self) -> (int, int):
        """
        :return: coordinates of random nonzero element (x, y) ~ (row, column)
        """
        index = np.random.randint(low=0, high=self.nonzero_count)
        return self.user_indices[index], self.item_indices[index]

    def __fit_preparation__(self, user_to_item: sp.csr_matrix, init_biases=False):
        """
        Initializes user-factorization and item-factorization matrices with values in range
        ( 0; 1 / sqrt(factors) )

        Also stores non zero elements in self.nonzero field

        :param user_to_item: User-to-Item sparse matrix with ratings (scipy.sparse.csr_matrix)
        :param init_biases: Responsible for initial biases (initializes them as mean)
        :return: None
        """
        np.random.seed(self.seed)

        n_users, n_items = user_to_item.shape
        k = self.factors

        self.user_factors = np.random.rand(n_users, k) / np.sqrt(k)
        self.item_factors = np.random.rand(n_items, k) / np.sqrt(k)

        self.user_indices, self.item_indices = user_to_item.nonzero()
        self.nonzero_count = user_to_item.count_nonzero()

        if init_biases:
            self.bias = user_to_item.mean()
            self.user_bias = np.array(user_to_item.mean(axis=1)).reshape(-1)
            self.item_bias = np.array(user_to_item.mean(axis=0)).reshape(-1)

    def predict(self, i, j):
        """
        Predicts relevance score of **item (j)** for **user (j)**

        :param i: user index
        :param j: item index
        :return:  score of user-item dot product embeddings
        """
        return self.user_factors[i] @ self.item_factors[j]

    def similar_items(self, item_id, top_k=10) -> (np.array, np.array):
        """
        Calculates similarity by Euclidean distance.

        :param item_id: zero indexed item id which corresponds to index of given user-to-item matrix for fit.
        :param top_k: number of required items
        :return: (item_ids_sorted_by_score, scores)
        """
        scores = np.linalg.norm(self.item_factors - self.item_factors[item_id], axis=1)
        indices = np.argsort(scores)
        return indices[:top_k], np.sort(scores)[:top_k]

    def recommend(self, user_index, amount=10) -> np.array:
        """
        :return user recommendations by **best** score, which user **hadn't been rated**.
        """
        user_predictions = self.user_factors[user_index] @ self.item_factors.T
        known_item_ids = self.item_indices[self.user_indices == user_index]
        user_predictions[known_item_ids] = np.NINF
        return np.argsort(user_predictions.reshape(-1))[-amount:]

    def rmse(self, user_to_item: sp.csr_matrix, tqdm_range=None, n_elements=250, use_all=False):
        if use_all:
            indices = range(self.nonzero_count)
        else:
            indices = np.random.randint(low=0, high=self.nonzero_count, size=n_elements)

        def get_error(idx):
            i, j = self.user_indices[idx], self.item_indices[idx]
            actual = user_to_item[i, j]
            return (self.predict(i, j) - actual) ** 2

        loss = np.fromiter(map(get_error, indices), dtype=np.float64)

        rmse_loss = np.sqrt(np.mean(loss))

        if tqdm_range is not None:
            tqdm_range.set_postfix_str("RMSE loss of random {} elements: {:.5f}".format(n_elements, rmse_loss))

        return rmse_loss
