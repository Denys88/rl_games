from enum import Enum


class Action(Enum):
    FOLD = 0
    CHECK_CALL = 1
    RAISE = 2
    ALL_IN = 3


class Stage(Enum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    END = 4
