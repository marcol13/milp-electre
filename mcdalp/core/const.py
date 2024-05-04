from .relations import PositivePreference, NegativePreference, Incomparible, Indifference
from enum import Enum

RELATIONS = [PositivePreference, NegativePreference, Incomparible, Indifference]
PARTIAL_OUTPUT = "outranking"
COMPLETE_OUTPUT = "p"

class RankingMode(Enum):
    PARTIAL = "partial"
    COMPLETE = "complete"

DEFAULT_SCORETABLE = {
    PositivePreference: {
        PositivePreference: 0,
        NegativePreference: 4,
        Indifference: 2,
        Incomparible: 3
    },
    NegativePreference: {
        PositivePreference: 4,
        NegativePreference: 0,
        Indifference: 2,
        Incomparible: 3
    },
    Indifference: {
        PositivePreference: 2,
        NegativePreference: 2,
        Indifference: 0,
        Incomparible: 2
    },
    Incomparible: {
        PositivePreference: 3,
        NegativePreference: 3,
        Indifference: 2,
        Incomparible: 0
    }
}