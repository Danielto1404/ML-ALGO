class EmptyLayerError(Exception):
    """
    Represent empty layer accessing.
    """

    def __init__(self):
        super().__init__("Unable to access empty layer.")
