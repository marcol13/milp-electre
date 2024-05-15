import pytest

from mcdalp.core.relations import PositivePreference, NegativePreference, Indifference, Incomparible

class TestRelations:
    def test_positivePreference(self):
        relation = PositivePreference(1, 0)
        assert relation.compatible(PositivePreference(1, 0)) == True
        assert relation.compatible(NegativePreference(1, 0)) == False
        assert relation.compatible(Indifference(1, 0)) == False
        assert relation.compatible(Incomparible(1, 0)) == False

        assert relation.compatible(PositivePreference(0, 1)) == False
        assert relation.compatible(NegativePreference(0, 1)) == True
        assert relation.compatible(Indifference(0, 1)) == False
        assert relation.compatible(Incomparible(0, 1)) == False

        assert relation.compatible(PositivePreference(1, 2)) == True
        assert relation.compatible(NegativePreference(1, 2)) == True
        assert relation.compatible(Indifference(1, 2)) == True
        assert relation.compatible(Incomparible(1, 2)) == True

    def test_negativePreference(self):
        relation = NegativePreference(1, 0)
        assert relation.compatible(PositivePreference(1, 0)) == False
        assert relation.compatible(NegativePreference(1, 0)) == True
        assert relation.compatible(Indifference(1, 0)) == False
        assert relation.compatible(Incomparible(1, 0)) == False

        assert relation.compatible(PositivePreference(0, 1)) == True
        assert relation.compatible(NegativePreference(0, 1)) == False
        assert relation.compatible(Indifference(0, 1)) == False
        assert relation.compatible(Incomparible(0, 1)) == False

        assert relation.compatible(PositivePreference(1, 2)) == True
        assert relation.compatible(NegativePreference(1, 2)) == True
        assert relation.compatible(Indifference(1, 2)) == True
        assert relation.compatible(Incomparible(1, 2)) == True

    def test_indifference(self):
        relation = Indifference(1, 0)
        assert relation.compatible(PositivePreference(1, 0)) == False
        assert relation.compatible(NegativePreference(1, 0)) == False
        assert relation.compatible(Indifference(1, 0)) == True
        assert relation.compatible(Incomparible(1, 0)) == False

        assert relation.compatible(PositivePreference(0, 1)) == False
        assert relation.compatible(NegativePreference(0, 1)) == False
        assert relation.compatible(Indifference(0, 1)) == True
        assert relation.compatible(Incomparible(0, 1)) == False

        assert relation.compatible(PositivePreference(1, 2)) == True
        assert relation.compatible(NegativePreference(1, 2)) == True
        assert relation.compatible(Indifference(1, 2)) == True
        assert relation.compatible(Incomparible(1, 2)) == True

    def test_incomparible(self):
        relation = Incomparible(1, 0)
        assert relation.compatible(PositivePreference(1, 0)) == False
        assert relation.compatible(NegativePreference(1, 0)) == False
        assert relation.compatible(Indifference(1, 0)) == False
        assert relation.compatible(Incomparible(1, 0)) == True

        assert relation.compatible(PositivePreference(0, 1)) == False
        assert relation.compatible(NegativePreference(0, 1)) == False
        assert relation.compatible(Indifference(0, 1)) == False
        assert relation.compatible(Incomparible(0, 1)) == True

        assert relation.compatible(PositivePreference(1, 2)) == True
        assert relation.compatible(NegativePreference(1, 2)) == True
        assert relation.compatible(Indifference(1, 2)) == True
        assert relation.compatible(Incomparible(1, 2)) == True
