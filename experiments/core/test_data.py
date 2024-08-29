from mcdalp.core.types import RankingModeType    
from itertools import product

test_thresholds = {
    "low": {
    "indifference": 0.05,
    "preference": 0.15,
    "veto": 0.25
},
    "medium": {
        "indifference": 0.15,
        "preference": 0.3,
        "veto": 0.5
    },
    "high": {
        "indifference": 0.25,
        "preference": 0.45,
        "veto": 0.75
    }
}

test_data = {
    "thresholds": test_thresholds.values(),
    "alternatives": list(range(8, 21, 2)),
    "criteria": list(range(3, 9)),
    "is_cost_threshold": [0.5],
    "mode": ["complete", "partial"],
    "all_results": [False],
    "binary_threshold": [0.5, 0.6, 0.7, 0.8],
}

def generate_test_data(iterate_keys: list[str], default_dict: dict, test_data: dict = test_data) -> list[dict]:
    if set(default_dict.keys()).intersection(set(iterate_keys)):
        raise ValueError("Iterate keys and default keys cannot have common keys")
    
    if any(isinstance(value, list) for value in default_dict.values()):
        raise ValueError("Default values cannot be of type list")
    
    values = [test_data[key] for key in iterate_keys]
    combinations = list(product(*values))
    
    dict_list = [dict(zip(iterate_keys, combination), **default_dict) for combination in combinations]

    return dict_list

    