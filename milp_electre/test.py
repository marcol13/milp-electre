from core.credibility import CredibilityMatrix
from core.relations import *
from core.score import Score
from outranking.crisp import CrispOutranking
import numpy as np

array = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
credibility = CredibilityMatrix(array)
print(credibility.matrix)

relation0 = PositivePreference(1, 0)
relation1 = NegativePreference(0, 1)
relation2 = IndifferenceRelation(1, 2)
relation3 = IncomparableRelation(1, 0)

print(relation0.compatible(relation1))
print(relation0.compatible(relation2))
print(relation0.compatible(relation3))

scoretable = {
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

score1 = Score(scoretable)
score1.show()

crisp = CrispOutranking(credibility, score1)
crisp.solve_partial()