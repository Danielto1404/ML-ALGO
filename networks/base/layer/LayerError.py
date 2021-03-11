class EmptyLayerError(Exception):
    """
    Represent empty layer accessing.
    """
    def __init__(self, message):
        super().__init__(message)
