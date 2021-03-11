class DecisionTree:
    def __init__(self, n_estimators, split_condition, split_function=None):
        self.n_estimators = n_estimators
        self.split_condition = split_condition
        self.split_function = split_function

    def fit(self):
        raise NotImplementedError("DecisionTree not implemented")

    def predict(self):
        raise NotImplementedError("DecisionTree not implemented")


class DecisionTreeRegression(DecisionTree):

    def fit(self):
        pass

    def predict(self):
        pass


class DecisionTreeClassifier(DecisionTree):

    def fit(self):
        pass

    def predict(self):
        pass


class StumpClassifier(DecisionTreeClassifier):
    def __init__(self, split_condition, split_function=None):
        super().__init__(n_estimators=1,
                         split_condition=split_condition,
                         split_function=split_function)


class StumpRegression(DecisionTreeRegression):
    def __init__(self, split_condition, split_function=None):
        super().__init__(n_estimators=1,
                         split_condition=split_condition,
                         split_function=split_function)
