import json

import numpy as np

def merge_all_data():
    files = ["experiments/data/electreIII/complete.json", "experiments/data/electreIII/partial.json", "experiments/data/prometheeI/partial.json", "experiments/data/prometheeII/complete.json", "experiments/data/netflow/completev2.json", "experiments/data/netflow/final_partial.json"]
    complete_files = ["experiments/data/electreIII/complete.json", "experiments/data/prometheeII/complete.json", "experiments/data/netflow/completev2.json"]
    partial_files = ["experiments/data/electreIII/partial.json", "experiments/data/prometheeI/partial.json", "experiments/data/netflow/final_partial.json"]
    crisp_files = ["experiments/data/netflow/final_partial.json", "experiments/data/netflow/completev2.json"]
    output_path = "experiments/data/crisp_aggregated_results.json"
    first_item = True

    with open(output_path, "w") as out:
        out.write("[\n")

        for file in crisp_files:
            with open(file, "r") as f:
                data = json.load(f)

                for entry in data:
                    if first_item:
                        first_item = False
                    else:
                        out.write(",\n")
                    json.dump(entry, out)

        out.write("\n]")

def get_aggregated_measure():
    data = []
    metric = "normalized_hit_ratio"
    with open("experiments/data/aggregated_results.json", "r") as f:
        for line in f:
            if line.startswith("[") or line.endswith("]"):
                continue
            if line.endswith(",\n"):
                line = line[:-2]
            line_data = json.loads(line)
            measure = list(map(lambda x: x[metric], line_data["measurements"]))
            data.extend(measure)

    data = np.array(data)

    print(f"Mean: {np.mean(data)}")
    print(f"Standard deviation: {np.std(data)}")

# def get_partial_measure():

def get_metrics_by_category():
    data = []
    metric = "rank_difference"
    with open("experiments/data/electreIII/partial.json", "r") as f:
        file_data = json.load(f)
        for line in file_data:
            if line["thresholds"]["indifference"] == 0.25:
                measure = list(map(lambda x: x[metric], line["measurements"]))
                data.extend(measure)

    data = np.array(data)

    print(f"Mean: {np.mean(data)}")
    print(f"Standard deviation: {np.std(data)}")

def get_metrics_by_threshold():
    data = []
    metric = "rank_difference"
    with open("experiments/data/crisp_aggregated_results.json", "r") as f:
        for line in f:
            if line.startswith("[") or line.endswith("]"):
                continue
            if line.endswith(",\n"):
                line = line[:-2]
            line_data = json.loads(line)
            if line_data["binary_threshold"] == 0.7:
                measure = list(map(lambda x: x[metric], line_data["measurements"]))
                data.extend(measure)

    data = np.array(data)

    print(f"Mean: {np.mean(data)}")
    print(f"Standard deviation: {np.std(data)}")

if __name__ == "__main__":
    # merge_all_data()
    # get_aggregated_measure()
    get_metrics_by_category()
    # get_metrics_by_threshold()