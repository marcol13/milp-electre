import numpy as np

class CredibilityMatrix:
    def __init__(self, matrix: np.ndarray):
        self.check_consistency(matrix)
        self.matrix = matrix

    def __is_array(self, matrix: np.ndarray):
        if not isinstance(matrix, np.ndarray):
            raise ValueError("Matrix must be a numpy array")
        
    def __is_square(self, matrix: np.ndarray):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square")
        
    def __is_normalized(self, matrix: np.ndarray):
        if not np.all(matrix >= 0) or not np.all(matrix <= 1):
            raise ValueError("Matrix must have values between 0 and 1")

    def check_consistency(self, matrix):
        try:
            self.__is_array(matrix)
            self.__is_square(matrix)
            self.__is_normalized(matrix)
        except ValueError:
            raise
