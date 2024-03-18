import numpy as np

def is_array(matrix: np.ndarray):
        if not isinstance(matrix, np.ndarray):
            raise ValueError("Matrix must be a numpy array")
        
def is_square(matrix: np.ndarray):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")
    
def is_normalized(matrix: np.ndarray):
    if not np.all(matrix >= 0) or not np.all(matrix <= 1):
        raise ValueError("Matrix must have values between 0 and 1")
    
def check_keys(dictionary, relations):
    if set(dictionary.keys()) != set(relations):
        raise ValueError(f"There are incorrect keys. Dictionary should includes: {[rel.name for rel in relations]}")