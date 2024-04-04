from .relations import PositivePreference, NegativePreference, Incomparible, Indifference

RELATIONS = [PositivePreference, NegativePreference, Incomparible, Indifference]

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