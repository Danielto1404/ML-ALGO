import numpy as np

from recsys.mf.bpr import BPR


class WARP(BPR):
    def __init__(self, iterations, factors, learning_rate, alpha, max_warp_sampled, seed):
        super().__init__(iterations, factors, learning_rate, alpha, seed)
        self.max_warp_sampled = max_warp_sampled

    def negative_choice(self, user):
        size = min(self.max_warp_sampled, len(self.negatives[user]))
        negatives = np.random.choice(self.negatives[user], size, replace=False)
        ids = dict(zip(np.arange(self.max_warp_sampled), negatives))
        scores = self.user_factors[user] @ self.item_factors[negatives].T
        best_id = np.argsort(scores)[-1]
        return ids[best_id]
