class KernelScaler:
    def __init__(self, value):
        self.value = value

    def scale(self, x):
        if self.value == 0:
            return float('inf')

        return x / self.value


class WindowScaler(KernelScaler):
    pass


class KNeighbourScaler(KernelScaler):
    pass
