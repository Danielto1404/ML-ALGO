from supervised.knn.base.KernelScaler import WindowScaler


class AdaptiveKNN:
    def __init__(self, kernel=None, kernel_scaler=WindowScaler(value=1)):
        self.kernel = kernel
        self.kernel_scaler = kernel_scaler

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
