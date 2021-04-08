class ROC:
    @staticmethod
    def auc(unbiased_values, n_positive, n_negative):
        """
        Calculates **AUC** and lists of **TPRs** and **FPRs** for different biases.

        Work's only with classes *+1* and *-1*

        For example:
            in **Logistic Regression** w0 is bias.
            in **SVM** w0 is bias.

        :param unbiased_values: g(xi, w)
        :param n_positive: actual amount of elements in positive class.
        :param n_negative: actual amount of elements in negative class.
        :return: ([TPR], [FPR], AUC)
        """
        TPR, FPR = [], []
        tpr, fpr, auc = 0, 0, 0
        for y in sorted(unbiased_values, reverse=True):
            if y == -1:
                fpr += 1 / n_negative
                FPR.append(fpr)
                auc += tpr * 1 / n_negative
            if y == 1:
                tpr += 1 / n_positive
                TPR.append(tpr)
            else:
                raise NotImplemented("Doesn't support auc functions for classes not equal to +-1")

        return TPR, FPR, auc

    @staticmethod
    def max_accuracy_threshold(probabilities, targets):
        eps = 1e-10
        sorted_probabilities, sorted_targets, thresholds = [], [], []

        for p, t in sorted(zip(probabilities, targets)):
            sorted_probabilities.append(p)
            sorted_targets.append(t)

        p_ = -eps
        max_prob = sorted_probabilities[-1]
        for p in sorted_probabilities + [max_prob + eps]:
            thresh = (p_ + p) / 2
            thresholds.append(thresh)
            p_ = p

        def accuracy(targets_, predicted_):
            accuracy_score = 0
            for et, ep in zip(targets_, predicted_):
                accuracy_score += int(et == ep)

        accuracies = [accuracy(targets_=targets,
                               predicted_=[1 if p > thresh else 0 for p in probabilities]) for thresh in thresholds]

        max_index, max_accuracy = -1, -1
        for i, a in accuracies:
            if a >= max_accuracy:
                max_accuracy = a
                max_index = i

        return thresholds[max_index]
