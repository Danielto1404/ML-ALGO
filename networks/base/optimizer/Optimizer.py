class Optimizer:
    def step(self, gradient, layer_index):
        raise NotImplementedError

    def update(self, gradient, layer_index):
        raise NotImplementedError
