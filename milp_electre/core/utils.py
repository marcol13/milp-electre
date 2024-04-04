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
        raise ValueError(f"There are incorrect keys. Dictionary should includes: {[str(rel) for rel in relations]}")
        
def get_numbers_relation(a: int, b: int):
    if a > b:
        return 1
    elif a < b:
        return -1
    else:
        return 0
    
def check_monotonicity(scoretable: dict, default_scoretable: dict):
    for a in scoretable.keys():
        for b in scoretable.keys():
            for c in scoretable.keys():
                if get_numbers_relation(scoretable[a][b], scoretable[a][c]) != get_numbers_relation(default_scoretable[a][b], default_scoretable[a][c]):
                    raise ValueError("Relations should be in the same order like in DEFAULT_SCORETABLE for all pairs")