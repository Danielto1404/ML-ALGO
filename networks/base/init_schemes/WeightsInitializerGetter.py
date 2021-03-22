from networks.base.init_schemes.Xavier import Xavier


class WeightsInitializerGetter:
    initializers = {
        'xavier': Xavier(),
    }

    @staticmethod
    def get(name):
        initializer = WeightsInitializerGetter.initializers.get(name)
        if initializer is None:
            raise UnknownWeightsInitializer(unknown_name=name,
                                            possible_names=list(WeightsInitializerGetter.initializers.keys()))

        return initializer


class UnknownWeightsInitializer(Exception):
    def __init__(self, unknown_name, possible_names):
        super(UnknownWeightsInitializer, self).__init__(
            """
                Unknown  weights initializer: {}
                Possible weights initializer: {}
            """.format(unknown_name, possible_names)
        )
