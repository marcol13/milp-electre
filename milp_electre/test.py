from core.credibility import CredibilityMatrix
from core.relations import *
import numpy as np

array = np.array([[0.5, 0.5], [0.5, 0.5]])
credibility = CredibilityMatrix(array)
print(credibility.matrix)

relation0 = PositivePreference(1, 0)
relation1 = NegativePreference(0, 1)
relation2 = IndifferenceRelation(1, 2)
relation3 = IncomparableRelation(1, 0)

print(relation0.compatible(relation1))
print(relation0.compatible(relation2))
print(relation0.compatible(relation3))