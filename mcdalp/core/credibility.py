import numpy as np
from .const import RELATIONS
from .utils import is_array, is_square, is_normalized, check_keys, is_sum_one, is_the_same_array


class CredibilityMatrix:
    def __init__(self, matrix: np.ndarray):
        self.matrix = np.asarray(matrix, dtype=int)
        self.check_consistency(self.matrix, matrix)
    
    def __len__(self):
        return self.matrix.shape[0]
    
    def __repr__(self) -> str:
        return f"{self.matrix}"

    def check_consistency(self, matrix, original_matrix):
        try:
            is_array(matrix)
            is_square(matrix)
            is_normalized(matrix)
            is_the_same_array(matrix, original_matrix)
        except ValueError:
            raise


class StochasticCredibilityMatrix():
    def __init__(self, matrices: np.ndarray):
        self.matrix = matrices
        for key in matrices:
            self.matrix[key] = np.asarray(matrices[key], dtype=float)
        self.check_consistency(self.matrix, matrices)

    def __len__(self):
        return self.matrix[RELATIONS[0]].shape[0]
        
    def __repr__(self) -> str:
        return f"{self.matrix}"

    def check_consistency(self, matrices, original_matrices):
        try:
            for matrix, org_matrix in zip(matrices.values(), original_matrices.values()):
                is_array(matrix)
                is_square(matrix)
                is_normalized(matrix)
                is_the_same_array(matrix, org_matrix)
            check_keys(matrices, RELATIONS)
            is_sum_one(matrices)
        except ValueError:
            raise
    

class ValuedCredibilityMatrix():
    def __init__(self, matrices: np.ndarray):
        self.matrix = np.asarray(matrices, dtype=float)
        self.check_consisntency(self.matrix, matrices)
        
    def __len__(self):
        self.matrix.shape[0]

    def __repr__(self) -> str:
        return f"{self.matrix}"

    def check_consistency(self, matrix, original_matrix):
        try:
            is_array(matrix)
            is_square(matrix)
            is_normalized(matrix)
            is_the_same_array(matrix, original_matrix)
        except ValueError:
            raise
