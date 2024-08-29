import pytest
from experiments.core.test_data import generate_test_data

class TestHelperFunctions():
    def test_generate_test_data(self):
        iterate_keys = ["alternatives", "criteria", "thresholds"]
        default_dict = {
            "mode": "complete",
            "all_results": False,
            "is_cost_threshold": 0.5
        }
        test_data = {
            "thresholds": [{"indifference": 0.05, "preference": 0.15, "veto": 0.25}],
            "alternatives": [8, 10, 12],
            "criteria": [3, 4, 5]
        }
        result = generate_test_data(iterate_keys, default_dict, test_data)
        assert len(result) == 9
        for r in result:
            assert r["mode"] == "complete"
            assert r["all_results"] == False
            assert r["is_cost_threshold"] == 0.5
            assert r["thresholds"] == {"indifference": 0.05, "preference": 0.15, "veto": 0.25}
            assert r["alternatives"] in [8, 10, 12]
            assert r["criteria"] in [3, 4, 5]

    def test_generate_test_data_with_common_keys(self):
        iterate_keys = ["alternatives", "criteria", "thresholds"]
        default_dict = {
            "mode": "complete",
            "all_results": False,
            "is_cost_threshold": 0.5,
            "criteria": 3
        }
        test_data = {
            "thresholds": [{"indifference": 0.05, "preference": 0.15, "veto": 0.25}],
            "alternatives": [8, 10, 12],
            "criteria": [3, 4, 5]
        }

        with pytest.raises(ValueError):
            _ = generate_test_data(iterate_keys, default_dict, test_data)

    def test_generate_test_data_with_list_default_values(self):
        iterate_keys = ["alternatives", "thresholds"]
        default_dict = {
            "mode": "complete",
            "all_results": False,
            "is_cost_threshold": 0.5,
            "criteria": [3, 5, 7]
        }
        test_data = {
            "thresholds": [{"indifference": 0.05, "preference": 0.15, "veto": 0.25}],
            "alternatives": [8, 10, 12],
            "criteria": [3, 4, 5]
        }

        with pytest.raises(ValueError):
            _ = generate_test_data(iterate_keys, default_dict, test_data)