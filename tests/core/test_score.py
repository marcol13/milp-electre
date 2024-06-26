import pytest
from mcdalp.core.relations import PositivePreference, NegativePreference, Indifference, Incomparible
from mcdalp.core.score import Score
from ..helpers import Dummy

class TestDistanceMatrix():
    def test_keysValidity(self):
        nonvalid_relation = {
            Dummy: {
                Dummy: 0,
                NegativePreference: 3,
                Indifference: 2,
                Incomparible: 3
            },
            NegativePreference: {
                Dummy: 4,
                NegativePreference: 0,
                Indifference: 2,
                Incomparible: 3
            },
            Indifference: {
                Dummy: 2,
                NegativePreference: 2,
                Indifference: 0,
                Incomparible: 2
            },
            Incomparible: {
                Dummy: 3,
                NegativePreference: 3,
                Indifference: 2,
                Incomparible: 0
            }
        }

        with pytest.raises(ValueError):
            _ = Score(nonvalid_relation)

    def test_lackOfKeys(self):
        lack_of_keys = {
            PositivePreference: {
                PositivePreference: 0,
                NegativePreference: 3,
                Indifference: 2
            },
            NegativePreference: {
                PositivePreference: 4,
                NegativePreference: 0,
                Indifference: 2
            },
            Indifference: {
                PositivePreference: 2,
                NegativePreference: 2,
                Indifference: 0
            }
        }

        with pytest.raises(ValueError):
            _ = Score(lack_of_keys)

    def test_selfRelation(self):
        self_relation = {
            PositivePreference: {
                PositivePreference: 1,
                NegativePreference: 4,
                Indifference: 2,
                Incomparible: 3
            },
            NegativePreference: {
                PositivePreference: 4,
                NegativePreference: 1,
                Indifference: 2,
                Incomparible: 3
            },
            Indifference: {
                PositivePreference: 2,
                NegativePreference: 2,
                Indifference: 1,
                Incomparible: 2
            },
            Incomparible: {
                PositivePreference: 3,
                NegativePreference: 3,
                Indifference: 2,
                Incomparible: 1
            }
        }

        with pytest.raises(ValueError):
            _ = Score(self_relation)

    def test_differentValues(self):
        different_values = {
            PositivePreference: {
                PositivePreference: 0,
                NegativePreference: 7,
                Indifference: 2,
                Incomparible: 3
            },
            NegativePreference: {
                PositivePreference: 6,
                NegativePreference: 0,
                Indifference: 2,
                Incomparible: 3
            },
            Indifference: {
                PositivePreference: 2,
                NegativePreference: 2,
                Indifference: 0,
                Incomparible: 2
            },
            Incomparible: {
                PositivePreference: 3,
                NegativePreference: 3,
                Indifference: 2,
                Incomparible: 0
            }
        }

        with pytest.raises(ValueError):
            _ = Score(different_values)

    def test_incorrectOrder(self):
        incorrect_order = {
            PositivePreference: {
                PositivePreference: 0,
                NegativePreference: 7,
                Indifference: 5,
                Incomparible: 3
            },
            NegativePreference: {
                PositivePreference: 7,
                NegativePreference: 0,
                Indifference: 5,
                Incomparible: 3
            },
            Indifference: {
                PositivePreference: 5,
                NegativePreference: 5,
                Indifference: 0,
                Incomparible: 5
            },
            Incomparible: {
                PositivePreference: 3,
                NegativePreference: 3,
                Indifference: 5,
                Incomparible: 0
            }
        }

        with pytest.raises(ValueError):
            _ = Score(incorrect_order)

    def test_validScore(self):
        valid_score = {
            PositivePreference: {
                PositivePreference: 0,
                NegativePreference: 7,
                Indifference: 1,
                Incomparible: 4
            },
            NegativePreference: {
                PositivePreference: 7,
                NegativePreference: 0,
                Indifference: 1,
                Incomparible: 4
            },
            Indifference: {
                PositivePreference: 1,
                NegativePreference: 1,
                Indifference: 0,
                Incomparible: 1
            },
            Incomparible: {
                PositivePreference: 4,
                NegativePreference: 4,
                Indifference: 1,
                Incomparible: 0
            }
        }

        try:
            _ = Score(valid_score)
        except ValueError:
            pytest.fail("Score raised ValueError unexpectedly")
