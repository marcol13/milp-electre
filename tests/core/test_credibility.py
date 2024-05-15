import pytest

from mcdalp.core.credibility import CredibilityMatrix, StochasticCredibilityMatrix, ValuedCredibilityMatrix

class TestCredibilityMatrix:
    def test_isMatrix(self):
        matrix = "test"

        with pytest.raises(ValueError):
            _ = CredibilityMatrix(matrix)

    def test_isSquare(self):
        matrix = [[0, 1], [1, 1], [1, 0]]

        with pytest.raises(ValueError):
            _ = CredibilityMatrix(matrix)

    def test_isNormalized(self):
        matrix = [[0, 2], [1, 1]]

        with pytest.raises(ValueError):
            _ = CredibilityMatrix(matrix)

    def test_isTheSameArray(self):
        matrix = [[0.5, 0.7], [0.3, 0]]

        with pytest.raises(ValueError):
            _ = CredibilityMatrix(matrix)

    def test_createCredibilityMatrix(self):
        matrix = [[0, 1, 0], [1, 0, 1], [1, 0, 0]]
        credibility_matrix = CredibilityMatrix(matrix)
        
        assert isinstance(credibility_matrix, CredibilityMatrix)

    def test_lenCredibilityMatrix(self):
        matrix = [[0, 1, 0], [1, 0, 1], [1, 0, 0]]
        credibility_matrix = CredibilityMatrix(matrix)

        assert len(credibility_matrix) == 3