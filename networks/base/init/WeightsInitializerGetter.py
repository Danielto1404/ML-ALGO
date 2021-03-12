from networks.base.init.Xavier import Xavier


class WeightsInitializerGetter:
    initializers = {
        'xavier': Xavier(),
    }

    @staticmethod
    def get(name):
        initializer = WeightsInitializerGetter.initializers.get(name)
        if initializer is None:
            raise UnknownWeightsInitializer(
                """
                Unknown  weights initializer: {}
                Possible weights initializer: {}
            """.format(name, list(WeightsInitializerGetter.initializers.keys())))

        return initializer


class UnknownWeightsInitializer(Exception):
    pass
