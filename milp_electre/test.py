from core.credibility import CredibilityMatrix, StochasticCredibilityMatrix, ValuedCredibilityMatrix
from core.relations import *
from core.score import Score
from outranking.crisp import CrispOutranking
from outranking.stochastic import StochasticOutranking
from outranking.valued import ValuedOutranking
import numpy as np
from core.visualize.hasse import Hasse

array = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
credibility = CredibilityMatrix(array)
print(credibility.matrix)

arr_pp = np.array([[0, 0.2, 0.1], [0, 0, 0], [0, 0.7, 0]], dtype=np.float32)
arr_pn = np.array([[0, 0.05, 0.2], [0, 0, 0.7], [0, 0.3, 0]], dtype=np.float32)
arr_i = np.array([[0, 0.65, 0], [0, 0, 0.2], [0.3, 0, 0]], dtype=np.float32)
arr_r = np.array([[0, 0.1, 0.7], [1, 0, 0.1], [0.7, 0, 0]], dtype=np.float32)
# stochastic_array = np.array(np.array(arr_pp, arr_pn, arr_i, arr_r))

stochastic_array = StochasticCredibilityMatrix({PositivePreference: arr_pp, NegativePreference: arr_pn, Indifference: arr_i, Incomparible: arr_r})

arr_p = np.array([[0, 1, 0.3], [0, 0, 0.1], [0.7, 0.1, 0]], dtype=np.float32)
valued_array = ValuedCredibilityMatrix(arr_p)

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

# crisp = CrispOutranking(credibility, score1)
# crisp.solve_partial()
# crisp.solve_complete()

# stochastic = StochasticOutranking(stochastic_array, score1)
# stochastic.solve_partial()
# stochastic.solve_complete()

valued = ValuedOutranking(valued_array, score1)

# valued.solve_partial()
valued.solve_complete()

new_array = np.array([[0,1,1],[0,0,1],[0,0,0]], dtype=np.int32)
bigger_array = np.array([[0,1,1,1], [0,0,1,1], [0,0,0,1], [0,0,0,0]], dtype=np.int32)
# cls = MyClass(new_array, ["A", "B", "C"])
# rel = get_relations(new_array, ["A", "B", "C"])
# hasse = generate_hasse_diagram(rel)
# print(rel)
# print(hasse)

# cls = MyClass([(1,2),(2,3),(1,3)])
# cls.draw_diagram()

# hasse = Hasse(new_array, ["A", "B", "C"])
hasse = Hasse(bigger_array, ["A", "B", "C", "D"])

# l.add_intent('o1', [0,1])
# l.add_intent('o2', [1,2])