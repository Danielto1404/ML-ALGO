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
                raise NotImplemented("Doesn't support auc function for classes not equal to +-1")

        return TPR, FPR, auc
