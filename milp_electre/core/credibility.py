import numpy as np
from core.const import RELATIONS
from core.utils import is_array, is_square, is_normalized, check_keys
from core.relations import PositivePreference, NegativePreference

# TODO: Make factory class
class CredibilityMatrix:
    def __init__(self, matrix: np.ndarray):
        self.check_consistency(matrix)
        self.matrix = matrix

    def get_size(self):
        return self.matrix.shape[0]

    def check_consistency(self, matrix):
        try:
            is_array(matrix)
            is_square(matrix)
            is_normalized(matrix)
            # TODO: check if it is integer matrix
        except ValueError:
            raise


class StochasticCredibilityMatrix():
    def __init__(self, matrices: np.ndarray):
        self.matrix = matrices
        
    def check_consistency(self, matrices):
        try:
            for matrix in matrices.values():
                is_array(matrix)
                is_square(matrix)
                is_normalized(matrix)
                # TODO: Check if it is float matrix
                # TODO: Check if it sum to 1
            check_keys(matrices, RELATIONS)
        except ValueError:
            raise

    def get_size(self):
        return self.matrix[RELATIONS[0]].shape[0]
    
class ValuedCredibilityMatrix():
    def __init__(self, matrices: np.ndarray):
        self.matrix = matrices
        
    def check_consistency(self, matrix):
        try:
            is_array(matrix)
            is_square(matrix)
            is_normalized(matrix)
            # TODO: Check if float 32
            # TODO: Check if min 0 and max 1
        except ValueError:
            raise

    def get_size(self):
        return self.matrix.shape[0]
    
    def get_reversed_matrix(self):
        return self.matrix.T
