class ObjectFactory:
    """General-purpose class to instantiate some other base class from rl_games. Usual use case it to instantiate algos, players etc.

    The ObjectFactory class is used to dynamically create any other object using a builder function (typically a lambda function).

    """

    def __init__(self):
        """Initialise a dictionary of builders with keys as `str` and values as functions.

        """
        self._builders = {}

    def register_builder(self, name, builder):
        """Register a passed builder by adding to the builders dict.

        Initialises runners and players for all algorithms available in the library using `rl_games.common.object_factory.ObjectFactory`

        Args:
            name (:obj:`str`): Key of the added builder.
            builder (:obj `func`): Function to return the requested object

        """
        self._builders[name] = builder

    def set_builders(self, builders):
        self._builders = builders
        
    def create(self, name, **kwargs):
        """Create the requested object by calling a registered builder function.

        Args:
            name (:obj:`str`): Key of the requested builder.
            **kwargs: Arbitrary kwargs needed for the builder function

        """
        builder = self._builders.get(name)
        if not builder:
            raise ValueError(name)
        return builder(**kwargs)