class EmptyLayerError(Exception):
    """
    Represent empty layers accessing.
    """

    def __init__(self):
        super().__init__("Unable to access empty layers.")
