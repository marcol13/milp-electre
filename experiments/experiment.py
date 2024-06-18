import numpy as np

from experiments.benchmarks.promethee_I import compare_promethee1
from experiments.benchmarks.promethee_II import compare_promethee2
from experiments.benchmarks.net_flow import compare_netflow_score
from experiments.plots import plot_heatmap

settings = {
        "alternatives": 8,
        "criteria": 5,
        "thresholds": {
            "indifference": 0.05,
            "preference": 0.15,
            "veto": 0.25
        },
        "is_cost_threshold": 0.5,
        "binary_threshold": 0.5,
        "mode": "partial",
        "all_results": False
    }

criteria_numbers = [3, 4, 5, 6, 7, 8]
alternatives_numbers = [6, 8, 10, 12, 14, 16, 18, 20]

kendall = []
nhr = []
rdm = []
for criteria in criteria_numbers:
    for alternatives in alternatives_numbers:
        settings["alternatives"] = alternatives
        settings["criteria"] = criteria
        metrics = compare_netflow_score(10, settings)
        metrics = np.average(metrics, axis=0)
        kendall.append(metrics[0])
        nhr.append(metrics[1])
        rdm.append(metrics[2])
        print(f"Criteria: {criteria}, Alternatives: {alternatives}")
        print(metrics)
        print("\n")

np.savetxt("kendall.csv", kendall, delimiter=",")
np.savetxt("nhr.csv", nhr, delimiter=",")
np.savetxt("rdm.csv", rdm, delimiter=",")

plot_heatmap(np.array(kendall).reshape(len(criteria_numbers), len(alternatives_numbers)), alternatives_numbers, criteria_numbers, "Number of alternatives", "Number of criteria", "Kendall tau measure for Net Flow Score")
plot_heatmap(np.array(nhr).reshape(len(criteria_numbers), len(alternatives_numbers)), alternatives_numbers, criteria_numbers, "Number of alternatives", "Number of criteria", "NHR for Net Flow Score")
plot_heatmap(np.array(rdm).reshape(len(criteria_numbers), len(alternatives_numbers)), alternatives_numbers, criteria_numbers, "Number of alternatives", "Number of criteria", "RDM for Net Flow Score")