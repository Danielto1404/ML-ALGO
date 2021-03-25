class EmptyLayerError(Exception):
    """
    Represent empty layers accessing.
    """

    def __init__(self, message="Unable to access empty layers."):
        super(EmptyLayerError, self).__init__(message)
