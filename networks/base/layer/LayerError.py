class EmptyLayerError(Exception):
    """
    Represent empty layer accessing.
    """

    def __init__(self, message):
        super().__init__(message)

    @staticmethod
    def raise_error():
        raise EmptyLayerError("Unable to access empty layer.")
