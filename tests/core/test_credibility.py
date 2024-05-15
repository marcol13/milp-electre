import pytest
import numpy as np

from mcdalp.core.credibility import CredibilityMatrix, StochasticCredibilityMatrix, ValuedCredibilityMatrix
from mcdalp.core.relations import PositivePreference, NegativePreference, Indifference, Incomparible

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

class TestStochasticCredibilityMatrix:
    def test_isMatrix(self):
        matrices = "test"

        with pytest.raises(TypeError):
            _ = StochasticCredibilityMatrix(matrices)

    def test_isSquare(self):
        matrices = {
            PositivePreference: np.array([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]]),
            NegativePreference: np.array([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]]),
            Indifference: np.array([[0.2, 0.1], [0.2, 0.1], [0.2, 0.1]]),
            Incomparible: np.array([[0.6, 0.5], [0.6, 0.5], [0.6, 0.5]])
        }

        with pytest.raises(ValueError):
            _ = StochasticCredibilityMatrix(matrices)

    def test_isNormalized(self):
        matrices = {
            PositivePreference: np.array([[0.1, 0.2, 1.5], [0.1, 0.2, 1.5], [0.1, 0.2, 1.5]]),
            NegativePreference: np.array([[0.1, 0.2, 0], [0.1, 0.2, 0], [0.1, 0.2, 0]]),
            Indifference: np.array([[0.2, 0.1, 0], [0.2, 0.1, 0], [0.2, 0.1, 0]]),
            Incomparible: np.array([[0.6, 0.5, 0], [0.6, 0.5, 0], [0.6, 0.5, 0]])
        }

        with pytest.raises(ValueError):
            _ = StochasticCredibilityMatrix(matrices)

    def test_isTheSameArray(self):
        matrices = {
            PositivePreference: np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]]),
            NegativePreference: np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0]]),
            Indifference: np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
            Incomparible: np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        }

        stochastic_credibility_matrix = StochasticCredibilityMatrix(matrices)
        assert isinstance(stochastic_credibility_matrix, StochasticCredibilityMatrix)

    def test_sumOne(self):
        matrices = {
            PositivePreference: np.array([[0.1, 0.2, 1], [0.1, 0.2, 1], [0.1, 0.2, 1]]),
            NegativePreference: np.array([[0.1, 0.2, 0], [0.1, 0.2, 0], [0.1, 0.2, 0]]),
            Indifference: np.array([[0.2, 0.1, 0], [0.2, 0.1, 0], [0.2, 0.1, 0]]),
            Incomparible: np.array([[0.7, 0.5, 0], [0.7, 0.5, 0], [0.7, 0.5, 0]])
        }

        with pytest.raises(ValueError):
            _ = StochasticCredibilityMatrix(matrices)

    def test_createStochasticCredibilityMatrix(self):
        matrices = {
            PositivePreference: np.array([[0.1, 0.2, 1], [0.1, 0.2, 1], [0.1, 0.2, 1]]),
            NegativePreference: np.array([[0.1, 0.2, 0], [0.1, 0.2, 0], [0.1, 0.2, 0]]),
            Indifference: np.array([[0.2, 0.1, 0], [0.2, 0.1, 0], [0.2, 0.1, 0]]),
            Incomparible: np.array([[0.6, 0.5, 0], [0.6, 0.5, 0], [0.6, 0.5, 0]])
        }
        stochastic_credibility_matrix = StochasticCredibilityMatrix(matrices)
        
        assert isinstance(stochastic_credibility_matrix, StochasticCredibilityMatrix)

    def test_lenStochasticCredibilityMatrix(self):
        matrices = {
            PositivePreference: np.array([[0.1, 0.2, 1], [0.1, 0.2, 1], [0.1, 0.2, 1]]),
            NegativePreference: np.array([[0.1, 0.2, 0], [0.1, 0.2, 0], [0.1, 0.2, 0]]),
            Indifference: np.array([[0.2, 0.1, 0], [0.2, 0.1, 0], [0.2, 0.1, 0]]),
            Incomparible: np.array([[0.6, 0.5, 0], [0.6, 0.5, 0], [0.6, 0.5, 0]])
        }
        stochastic_credibility_matrix = StochasticCredibilityMatrix(matrices)

        assert len(stochastic_credibility_matrix) == 3