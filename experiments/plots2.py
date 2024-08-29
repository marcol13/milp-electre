import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from collections import defaultdict
import matplotlib as mpl
import pandas as pd

def plot_heatmap(matrix: np.array, xticks_label: list[str], yticks_label: list[str], xlabel: str, ylabel: str, title: str, save_file: str, figsize: tuple = (13, 6), vmin: float = 0, vmax: float = 1):
    fig, ax = plt.subplots(figsize=figsize)

    min_val = np.min(matrix)
    max_val = np.max(matrix)

    if vmin == -1:
        min_scaled = (min_val + 1) / 2
        max_scaled = (max_val + 1) / 2
    else:
        min_scaled = min_val
        max_scaled = max_val

    # cmap = (mpl.colors.ListedColormap(['white', 'black', 'white']))
    colors = [
        (0, 'white'),                 # Biały od vmin do data_min
        (max(0, min_scaled - 0.0001), 'white'),
        (max(0, min_scaled - 0.0001), plt.cm.Wistia(0.0)),  # Początek Wistia od data_min
        (min(1, max_scaled + 0.0001), plt.cm.Wistia(1.0)),  # Koniec Wistia do data_max
        (min(1, max_scaled + 0.0001), 'white'),             # Biały od data_max do vmax
        (1, 'white')
    ]
    print(vmin)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# Tworzenie niestandardowej mapy kolorów
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)
    # dmap = plt.cm.ScalarMappable(cmap='Wistia')

    cax = ax.matshow(matrix, cmap=cmap, norm=norm)
    dax = ax.matshow(matrix, cmap='Wistia')
    dbar = fig.colorbar(dax, ax=ax)
    cbar = fig.colorbar(cax, ax=ax, ticks=[vmin, np.min(matrix), np.max(matrix), vmax])
    
    dbar.ax.tick_params(labelsize=16)
    cbar.ax.tick_params(labelsize=16)
    # fig.colorbar(mpl.cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap='magma'), boundaries=[np.min(matrix), np.max(matrix)])
    # fig.colorbar(cax, ticks=[0, np.min(matrix), np.max(matrix), 1], boundaries=[0, 1])

    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_xticklabels([''] + xticks_label, fontsize=16)
    ax.set_yticklabels([''] + yticks_label, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

    ax.set_title(title)

    # plt.show()
    plt.savefig(save_file)

def get_avg(data: list[dict], measure: str, data_key: str):
    result = []
    for survey in data:
        obj = {
            "alternatives": survey["alternatives"],
            data_key: survey[data_key],
            "measure": np.average([entry[measure] for entry in survey["measurements"]])
        }
        result.append(obj)
    
    return result

def get_median(data: list[dict], measure: str, is_time: bool = False):
    result = []
    for survey in data:
        obj = {
            "alternatives": survey["alternatives"],
            "measure": np.median([entry[measure] for entry in survey["measurements"]])
        }
        result.append(obj)
    
    return result

def as_matrix(data: list[dict], columns, rows):
    matrix = np.zeros((len(rows), len(columns)))
    for entry in data:
        i = rows.index(entry["criteria"])
        j = columns.index(entry["alternatives"])
        matrix[i, j] = entry["measure"]
    return matrix

def plot_pivot(data_comp: pd.DataFrame, data_lp: pd.DataFrame, save_filename: str):
    data_comp_criteria = data_comp.loc[[4, 6, 8]]
    data_lp_criteria = data_lp.loc[[4, 6, 8]]

    plt.figure(figsize=(13, 6))

    plt.plot(data_comp_criteria.columns, data_comp_criteria.loc[4], label="4 criteria", color='blue')
    plt.plot(data_comp_criteria.columns, data_comp_criteria.loc[6], label="6 criteria", color='green')
    plt.plot(data_comp_criteria.columns, data_comp_criteria.loc[8], label="8 criteria", color='red')

    plt.plot(data_lp_criteria.columns, data_lp_criteria.loc[4], linestyle='dashed', color='blue')
    plt.plot(data_lp_criteria.columns, data_lp_criteria.loc[6], linestyle='dashed', color='green')
    plt.plot(data_lp_criteria.columns, data_lp_criteria.loc[8], linestyle='dashed', color='red')

    plt.xlabel("Number of alternatives", fontsize=20)
    plt.ylabel("Time [s]", fontsize=20)
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='4 criteria'),
        Line2D([0], [0], color='green', lw=2, label='6 criteria'),
        Line2D([0], [0], color='red', lw=2, label='8 criteria'),
        Line2D([0], [0], color='white', lw=0, label=''),
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='Comparison Method'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Linear Programming')
    ]

    # Adding the custom legend to the plot
    plt.legend(handles=legend_elements, loc='best', fontsize=18)
    # plt.show()
    plt.savefig(save_filename)

def plot_line_chart(data1, data2, compare_method_str: str, save_filename: str):
    plt.figure(figsize=(13, 6))

    X1 = [entry["alternatives"] for entry in data1]
    Y1 = [entry["measure"] for entry in data1]

    X2 = [entry["alternatives"] for entry in data2]
    Y2 = [entry["measure"] for entry in data2]

    plt.plot(X1, Y1, label=compare_method_str)
    plt.plot(X2, Y2, label="Linear Programming")

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Number of alternatives", fontsize=20)
    plt.ylabel("Time [s]", fontsize=20)
    plt.legend(loc='best', fontsize=18)
    plt.savefig(save_filename)


def plot_whole_times(filename: str, save_filename: str, compare_method_str: str):
    plt.clf()
    with open(filename, "r") as f:
        data = json.load(f)
        data = map(lambda x: {**x, "level": "low" if x["thresholds"]["indifference"] == 0.05 else "mid" if x["thresholds"]["indifference"] == 0.15 else "high"}, data)
        data = list(data)
        measure_dict = defaultdict(list)
        measure_data = []

        for survey in data:
            measure_dict[survey["alternatives"]].extend(survey["measurements"])

        for key, value in measure_dict.items():
            measure_data.append({
                "alternatives": key,
                "measurements": value
            })

        median_time_comp = get_avg(measure_data, "time_comp", "alternatives")
        median_time_lp = get_avg(measure_data, "time_lp", "alternatives")

        sorted_median_time_comp = sorted(median_time_comp, key=lambda x: x["alternatives"])
        sorted_median_time_lp = sorted(median_time_lp, key=lambda x: x["alternatives"])

        plot_line_chart(sorted_median_time_comp, sorted_median_time_lp, compare_method_str, save_filename)

def plot_whole_data_heatmap(filename: str, data_key: str, measure: str, save_filename: str):
    plt.clf()
    with open(filename, "r") as f:
        data = json.load(f)
        data = map(lambda x: {**x, "level": "low" if x["thresholds"]["indifference"] == 0.05 else "mid" if x["thresholds"]["indifference"] == 0.15 else "high"}, data)
        data = list(data)
        measure_dict = defaultdict(list)
        measure_data = []

        for survey in data:
            measure_dict[(survey["alternatives"], survey[data_key])].extend(survey["measurements"])

        for key, value in measure_dict.items():
            measure_data.append({
                "alternatives": key[0],
                data_key: key[1],
                "measurements": value
            })

        measure_data = get_avg(measure_data, measure, data_key)

        if measure == "rank_difference":
            measure_data = list(map(lambda x: {**x, "measure": 1 - x["measure"]}, measure_data))

        df = pd.DataFrame(measure_data)
        pivot_df = df.pivot(index=data_key, columns='alternatives', values='measure')

        y_ticks = ["low", "mid", "high"] if data_key == "level" else [3, 4, 5, 6, 7, 8] if data_key == "criteria" else [0.5, 0.6, 0.7, 0.8]
        y_label = "Thresholds q, p, v" if data_key == "level" else "Number of criteria" if data_key == "criteria" else "Cutting threshold"
        figsize = (13, 6) if data_key == "criteria" else (18, 5)
        vmin = -1 if measure == "kendall_tau" else 0
        vmax = 1
        # plot_heatmap(pivot_df, [6, 8, 10, 12, 14, 16, 18, 20], [3, 4, 5, 6, 7, 8], "Number of alternatives", "Number of criteria", "", save_filename)
        plot_heatmap(pivot_df, [6, 8, 10, 12, 14, 16, 18, 20], y_ticks, "Number of alternatives", y_label, "", save_filename, figsize, vmin=vmin, vmax=vmax)

# def partial_plots(filename: str, base_folder: str):
#     # Thresholds plots
#     plot_heatmap_by_threshold(filename, 0.05, "normalized_rank_difference", f"{base_folder}/nrd_005.png")
#     plot_heatmap_by_threshold(filename, 0.15, "normalized_rank_difference", f"{base_folder}/nrd_015.png")
#     plot_heatmap_by_threshold(filename, 0.25, "normalized_rank_difference", f"{base_folder}/nrd_025.png")

#     plot_heatmap_by_threshold(filename, 0.05, "normalized_hit_ratio", f"{base_folder}/nhr_005.png")
#     plot_heatmap_by_threshold(filename, 0.15, "normalized_hit_ratio", f"{base_folder}/nhr_015.png")
#     plot_heatmap_by_threshold(filename, 0.25, "normalized_hit_ratio", f"{base_folder}/nhr_025.png")

#     plot_heatmap_by_threshold(filename, 0.05, "rank_difference", f"{base_folder}/rd_005.png")
#     plot_heatmap_by_threshold(filename, 0.15, "rank_difference", f"{base_folder}/rd_015.png")
#     plot_heatmap_by_threshold(filename, 0.25, "rank_difference", f"{base_folder}/rd_025.png")

#     # Criteria plots
#     plot_heatmap_by_criteria(filename, 4, "normalized_rank_difference", f"{base_folder}/nrd_criteria_4.png")
#     plot_heatmap_by_criteria(filename, 6, "normalized_rank_difference", f"{base_folder}/nrd_criteria_6.png")
#     plot_heatmap_by_criteria(filename, 8, "normalized_rank_difference", f"{base_folder}/nrd_criteria_8.png")

#     plot_heatmap_by_criteria(filename, 4, "normalized_hit_ratio", f"{base_folder}/nhr_criteria_4.png")
#     plot_heatmap_by_criteria(filename, 6, "normalized_hit_ratio", f"{base_folder}/nhr_criteria_6.png")
#     plot_heatmap_by_criteria(filename, 8, "normalized_hit_ratio", f"{base_folder}/nhr_criteria_8.png")

#     plot_heatmap_by_criteria(filename, 4, "rank_difference", f"{base_folder}/rd_criteria_4.png")
#     plot_heatmap_by_criteria(filename, 6, "rank_difference", f"{base_folder}/rd_criteria_6.png")
#     plot_heatmap_by_criteria(filename, 8, "rank_difference", f"{base_folder}/rd_criteria_8.png")

#     # Time plots
#     plot_times_by_threshold(filename, 0.05, f"{base_folder}/time_005.png")
#     plot_times_by_threshold(filename, 0.15, f"{base_folder}/time_015.png")
#     plot_times_by_threshold(filename, 0.25, f"{base_folder}/time_025.png")


def create_complete_plots(filename: str, base_folder: str, compare_method: str, representation: str):
    plot_whole_data_heatmap(filename, "criteria", "kendall_tau", f"{base_folder}/kendall_criteria.png")
    plot_whole_data_heatmap(filename, "level", "kendall_tau", f"{base_folder}/kendall_level.png")

    plot_whole_data_heatmap(filename, "criteria", "normalized_hit_ratio", f"{base_folder}/nhr_criteria.png")
    plot_whole_data_heatmap(filename, "level", "normalized_hit_ratio", f"{base_folder}/nhr_level.png")

    plot_whole_data_heatmap(filename, "criteria", "rank_difference", f"{base_folder}/rd_criteria.png")
    plot_whole_data_heatmap(filename, "level", "rank_difference", f"{base_folder}/rd_level.png")

    if representation == "crisp":
        plot_whole_data_heatmap(filename, "binary_threshold", "kendall_tau", f"{base_folder}/kendall_threshold.png")
        plot_whole_data_heatmap(filename, "binary_threshold", "normalized_hit_ratio", f"{base_folder}/nhr_threshold.png")
        plot_whole_data_heatmap(filename, "binary_threshold", "rank_difference", f"{base_folder}/rd_threshold.png")

    plot_whole_times(filename, f"{base_folder}/time.png", compare_method)

def create_partial_plots(filename: str, base_folder: str, compare_method: str, representation: str):
    plot_whole_data_heatmap(filename, "criteria", "normalized_rank_difference", f"{base_folder}/nrd_criteria.png")
    plot_whole_data_heatmap(filename, "level", "normalized_rank_difference", f"{base_folder}/nrd_level.png")

    plot_whole_data_heatmap(filename, "criteria", "normalized_hit_ratio", f"{base_folder}/nhr_criteria.png")
    plot_whole_data_heatmap(filename, "level", "normalized_hit_ratio", f"{base_folder}/nhr_level.png")

    plot_whole_data_heatmap(filename, "criteria", "rank_difference", f"{base_folder}/rd_criteria.png")
    plot_whole_data_heatmap(filename, "level", "rank_difference", f"{base_folder}/rd_level.png")

    if representation == "crisp":
        plot_whole_data_heatmap(filename, "binary_threshold", "normalized_rank_difference", f"{base_folder}/nrd_threshold.png")
        plot_whole_data_heatmap(filename, "binary_threshold", "normalized_hit_ratio", f"{base_folder}/nhr_threshold.png")
        plot_whole_data_heatmap(filename, "binary_threshold", "rank_difference", f"{base_folder}/rd_threshold.png")

    plot_whole_times(filename, f"{base_folder}/time.png", compare_method)


def outranking_complete_plot():
    filename = "experiments/data/netflow/completev2.json"
    base_folder = "experiments/visual/final/outranking/complete"

    create_complete_plots(filename, base_folder, "Net Flow Score", "crisp")

def outranking_partial_plot():
    filename = "experiments/data/netflow/final_partial.json"
    base_folder = "experiments/visual/final/outranking/partial"

    create_partial_plots(filename, base_folder, "Positive and negative flows", "crisp")

def electre_complete_plot():
    filename = "experiments/data/electreIII/complete.json"
    base_folder = "experiments/visual/final/electre/complete"

    create_complete_plots(filename, base_folder, "Distillation + rank", "valued")

def electre_partial_plot():
    filename = "experiments/data/electreIII/partial.json"
    base_folder = "experiments/visual/final/electre/partial"

    create_partial_plots(filename, base_folder, "Distillation", "valued")

def promethee_complete_plot():
    filename = "experiments/data/prometheeII/complete.json"
    base_folder = "experiments/visual/final/promethee/complete"

    create_complete_plots(filename, base_folder, "Net Flow Score", "valued")

def promethee_partial_plot():
    filename = "experiments/data/prometheeI/partial.json"
    base_folder = "experiments/visual/final/promethee/partial"

    create_partial_plots(filename, base_folder, "Positive and negative flows", "valued")

if __name__ == "__main__":
    # plot_crisp_heatmap_by_threshold("experiments/data/netflow/completev2.json", 0.05, "rank_difference", "experiments/visual/outranking/complete/rd_005.png")
    # plot_crisp_heatmap_by_criteria("experiments/data/netflow/completev2.json", 4, "rank_difference", "experiments/visual/outranking/complete/rd_criteria_4.png")
    # plot_whole_data_heatmap("experiments/data/netflow/completev2.json", "criteria", "kendall_tau", "experiments/visual/outranking/complete/nrd_level.png")
    # plot_whole_times("experiments/data/netflow/completev2.json", "experiments/visual/outranking/complete/time.png", "Net Flow Score")

    outranking_complete_plot()
    outranking_partial_plot()
    electre_complete_plot()
    electre_partial_plot()
    promethee_complete_plot()
    promethee_partial_plot()