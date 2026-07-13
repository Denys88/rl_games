import numpy as np
from enums import Action


class AlwaysCallPlayer:
    def __call__(self, _):
        return Action.CHECK_CALL


class AlwaysAllInPlayer:
    def __call__(self, _):
        return Action.ALL_IN


class RandomPlayer:
    def __call__(self, _):
        return np.random.choice(
            [Action.FOLD, Action.CHECK_CALL, Action.RAISE, Action.ALL_IN]
        )
