from benchmarks.promethee_I import compare_promethee1

settings = {
    "alternatives": 8,
    "criteria": 5,
    "thresholds": {
        "indifference": 0.05,
        "preference": 0.15,
        "veto": 0.25
    },
    "is_cost_threshold": 0.5,
    "mode": "partial",
    "all_results": True
}

metrics = compare_promethee1(10, settings)
print(metrics)