from core.credibility import CredibilityMatrix
import numpy as np

array = np.array([[0.5, 0.5], [0.5, 0.5]])
credibility = CredibilityMatrix(array)
print(credibility.matrix)