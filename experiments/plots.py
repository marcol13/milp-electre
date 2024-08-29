import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from collections import defaultdict
import matplotlib as mpl
import pandas as pd

def plot_heatmap(matrix: np.array, xticks_label: list[str], yticks_label: list[str], xlabel: str, ylabel: str, title: str, save_file: str, figsize: tuple = (13, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    # cmap = (mpl.colors.ListedColormap(['white', 'black', 'white']))
    colors = [
        (0, 'white'),                 # Biały od vmin do data_min
        (max(0, np.min(matrix) - 0.01), 'white'),
        (max(0, np.min(matrix) - 0.01), plt.cm.Wistia(0.0)),  # Początek Wistia od data_min
        (min(1, np.max(matrix) + 0.01), plt.cm.Wistia(1.0)),  # Koniec Wistia do data_max
        (min(1, np.max(matrix) + 0.01), 'white'),             # Biały od data_max do vmax
        (1, 'white')
    ]
    norm = mcolors.Normalize(vmin=0, vmax=1)

# Tworzenie niestandardowej mapy kolorów
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)
    # dmap = plt.cm.ScalarMappable(cmap='Wistia')

    cax = ax.matshow(matrix, cmap=cmap, norm=norm)
    dax = ax.matshow(matrix, cmap='Wistia')
    dbar = fig.colorbar(dax, ax=ax)
    cbar = fig.colorbar(cax, ax=ax, ticks=[0, np.min(matrix), np.max(matrix), 1])
    
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

def get_avg(data: list[dict], measure: str, is_criteria: bool = False):
    result = []
    if is_criteria:
        for survey in data:
            obj = {
                "alternatives": survey["alternatives"],
                "level": survey["level"],
                "measure": np.average([entry[measure] for entry in survey["measurements"]])
            }
            result.append(obj)
    else: 
        for survey in data:
            obj = {
                "alternatives": survey["alternatives"],
                "criteria": survey["criteria"],
                "measure": np.average([entry[measure] for entry in survey["measurements"]])
            }
            result.append(obj)
    return result

def get_median(data: list[dict], measure: str, is_criteria: bool = False):
    result = []
    if is_criteria:
        for survey in data:
            obj = {
                "alternatives": survey["alternatives"],
                "level": survey["level"],
                "measure": np.median([entry[measure] for entry in survey["measurements"]])
            }
            result.append(obj)
    else:
        for survey in data:
            obj = {
                "alternatives": survey["alternatives"],
                "criteria": survey["criteria"],
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

def plot_times_by_threshold(filename: str, threshold_q: float, save_filename: str):
    plt.clf()
    with open(filename, "r") as f:
        data = json.load(f)
        th_data = filter(lambda x: x["thresholds"]["indifference"] == threshold_q, data)
        th_data2 = filter(lambda x: x["thresholds"]["indifference"] == threshold_q, data)

        time_comp = pd.DataFrame(get_median(th_data, "time_comp"))
        time_lp = pd.DataFrame(get_median(th_data2, "time_lp"))

        time_comp_pivot = time_comp.pivot(index='criteria', columns='alternatives', values='measure')
        time_lp_pivot = time_lp.pivot(index='criteria', columns='alternatives', values='measure')

        # print(time_comp_pivot)
        # print(time_lp_pivot)
        plot_pivot(time_comp_pivot, time_lp_pivot, save_filename)

# def plot_times_by_criteria(filename: str, criteria: float, save_filename: str):
#     plt.clf()
#     with open(filename, "r") as f:
#         data = json.load(f)
#         data = map(lambda x: {**x, "level": "low" if x["thresholds"]["indifference"] == 0.05 else "mid" if x["thresholds"]["indifference"] == 0.15 else "high"}, data)
#         th_data = filter(lambda x: x["criteria"] == criteria, data)
#         th_data2 = filter(lambda x: x["criteria"] == criteria, data)

#         time_comp = pd.DataFrame(get_median(th_data, "time_comp", True))
#         time_lp = pd.DataFrame(get_median(th_data2, "time_lp", True))

#         time_comp_pivot = time_comp.pivot(index='level', columns='alternatives', values='measure')
#         time_lp_pivot = time_lp.pivot(index='level', columns='alternatives', values='measure')

#         # print(time_comp_pivot)
#         # print(time_lp_pivot)
#         plot_pivot(time_comp_pivot, time_lp_pivot, save_filename)


def plot_heatmap_by_threshold(filename: str, threshold_q: float, measure: str, save_filename: str):
    plt.clf()
    with open(filename, "r") as f:
        data = json.load(f)
        th_data = filter(lambda x: x["thresholds"]["indifference"] == threshold_q, data)

        measure_data = get_avg(th_data, measure)

        if measure == "rank_difference":
            measure_data = list(map(lambda x: {**x, "measure": 1 - x["measure"]}, measure_data))

        df = pd.DataFrame(measure_data)
        pivot_df = df.pivot(index='criteria', columns='alternatives', values='measure')

        plot_heatmap(pivot_df, [6, 8, 10, 12, 14, 16, 18, 20], [3, 4, 5, 6, 7, 8], "Number of alternatives", "Number of criteria", "", save_filename)

def plot_heatmap_by_criteria(filename: str, criteria: int, measure: str, save_filename: str):
    plt.clf()
    with open(filename, "r") as f:
        data = json.load(f)
        data = map(lambda x: {**x, "level": "low" if x["thresholds"]["indifference"] == 0.05 else "mid" if x["thresholds"]["indifference"] == 0.15 else "high"}, data)
        data = list(data)
        th_data = filter(lambda x: x["criteria"] == criteria, data)

        measure_data = get_avg(th_data, measure, True)

        if measure == "rank_difference":
            measure_data = list(map(lambda x: {**x, "measure": 1 - x["measure"]}, measure_data))

        df = pd.DataFrame(measure_data)
        pivot_df = df.pivot(index='level', columns='alternatives', values='measure')

        plot_heatmap(pivot_df, [6, 8, 10, 12, 14, 16, 18, 20], ["low", "mid", "high"], "Number of alternatives", "Thresholds q, p, v", "", save_filename, (18, 5))

def plot_crisp_heatmap_by_threshold(filename: str, threshold_q: float, measure: str, save_filename: str):
    plt.clf()
    with open(filename, "r") as f:
        data = json.load(f)
        measure_dict = defaultdict(list)
        measure_data = []
        th_data = list(filter(lambda x: x["thresholds"]["indifference"] == threshold_q, data))

        for survey in th_data:
            measure_dict[(survey["alternatives"], survey["criteria"])].extend(survey["measurements"])

        for key, value in measure_dict.items():
            measure_data.append({
                "alternatives": key[0],
                "criteria": key[1],
                "measurements": value
            })

        measure_data = get_avg(measure_data, measure)

        if measure == "rank_difference":
            measure_data = list(map(lambda x: {**x, "measure": 1 - x["measure"]}, measure_data))

        df = pd.DataFrame(measure_data)
        pivot_df = df.pivot(index='criteria', columns='alternatives', values='measure')

        plot_heatmap(pivot_df, [6, 8, 10, 12, 14, 16, 18, 20], [3, 4, 5, 6, 7, 8], "Number of alternatives", "Number of criteria", "", save_filename)

def plot_crisp_heatmap_by_criteria(filename: str, criteria: int, measure: str, save_filename: str):
    plt.clf()
    with open(filename, "r") as f:
        data = json.load(f)
        data = map(lambda x: {**x, "level": "low" if x["thresholds"]["indifference"] == 0.05 else "mid" if x["thresholds"]["indifference"] == 0.15 else "high"}, data)
        data = list(data)
        measure_dict = defaultdict(list)
        measure_data = []
        th_data = list(filter(lambda x: x["criteria"] == criteria, data))

        for survey in th_data:
            measure_dict[(survey["alternatives"], survey["level"])].extend(survey["measurements"])

        for key, value in measure_dict.items():
            measure_data.append({
                "alternatives": key[0],
                "level": key[1],
                "measurements": value
            })

        measure_data = get_avg(measure_data, measure, True)

        if measure == "rank_difference":
            measure_data = list(map(lambda x: {**x, "measure": 1 - x["measure"]}, measure_data))

        df = pd.DataFrame(measure_data)
        pivot_df = df.pivot(index='level', columns='alternatives', values='measure')

        # plot_heatmap(pivot_df, [6, 8, 10, 12, 14, 16, 18, 20], [3, 4, 5, 6, 7, 8], "Number of alternatives", "Number of criteria", "", save_filename)
        plot_heatmap(pivot_df, [6, 8, 10, 12, 14, 16, 18, 20], ["low", "mid", "high"], "Number of alternatives", "Thresholds q, p, v", "", save_filename, (18, 5))

def plot_crisp_heatmap_by_binary_threshold(filename: str, measure: str, save_filename: str):
    plt.clf()
    with open(filename, "r") as f:
        data = json.load(f)
        data = map(lambda x: {**x, "level": "low" if x["thresholds"]["indifference"] == 0.05 else "mid" if x["thresholds"]["indifference"] == 0.15 else "high"}, data)
        data = list(data)
        measure_dict = defaultdict(list)
        measure_data = []
        # th_data = list(filter(lambda x: x["binary_threshold"] == binary_threshold, data))

        for survey in data:
            measure_dict[(survey["alternatives"], survey["binary_threshold"])].extend(survey["measurements"])

        for key, value in measure_dict.items():
            measure_data.append({
                "alternatives": key[0],
                "binary_threshold": key[1],
                "measurements": value
            })

        measure_data = get_avg(measure_data, measure, True)

        if measure == "rank_difference":
            measure_data = list(map(lambda x: {**x, "measure": 1 - x["measure"]}, measure_data))

        df = pd.DataFrame(measure_data)
        pivot_df = df.pivot(index='level', columns='alternatives', values='measure')

        # plot_heatmap(pivot_df, [6, 8, 10, 12, 14, 16, 18, 20], [3, 4, 5, 6, 7, 8], "Number of alternatives", "Number of criteria", "", save_filename)
        plot_heatmap(pivot_df, [6, 8, 10, 12, 14, 16, 18, 20], ["low", "mid", "high"], "Number of alternatives", "Thresholds q, p, v", "", save_filename, (18, 5))

def partial_plots(filename: str, base_folder: str):
    # Thresholds plots
    plot_heatmap_by_threshold(filename, 0.05, "normalized_rank_difference", f"{base_folder}/nrd_005.png")
    plot_heatmap_by_threshold(filename, 0.15, "normalized_rank_difference", f"{base_folder}/nrd_015.png")
    plot_heatmap_by_threshold(filename, 0.25, "normalized_rank_difference", f"{base_folder}/nrd_025.png")

    plot_heatmap_by_threshold(filename, 0.05, "normalized_hit_ratio", f"{base_folder}/nhr_005.png")
    plot_heatmap_by_threshold(filename, 0.15, "normalized_hit_ratio", f"{base_folder}/nhr_015.png")
    plot_heatmap_by_threshold(filename, 0.25, "normalized_hit_ratio", f"{base_folder}/nhr_025.png")

    plot_heatmap_by_threshold(filename, 0.05, "rank_difference", f"{base_folder}/rd_005.png")
    plot_heatmap_by_threshold(filename, 0.15, "rank_difference", f"{base_folder}/rd_015.png")
    plot_heatmap_by_threshold(filename, 0.25, "rank_difference", f"{base_folder}/rd_025.png")

    # Criteria plots
    plot_heatmap_by_criteria(filename, 4, "normalized_rank_difference", f"{base_folder}/nrd_criteria_4.png")
    plot_heatmap_by_criteria(filename, 6, "normalized_rank_difference", f"{base_folder}/nrd_criteria_6.png")
    plot_heatmap_by_criteria(filename, 8, "normalized_rank_difference", f"{base_folder}/nrd_criteria_8.png")

    plot_heatmap_by_criteria(filename, 4, "normalized_hit_ratio", f"{base_folder}/nhr_criteria_4.png")
    plot_heatmap_by_criteria(filename, 6, "normalized_hit_ratio", f"{base_folder}/nhr_criteria_6.png")
    plot_heatmap_by_criteria(filename, 8, "normalized_hit_ratio", f"{base_folder}/nhr_criteria_8.png")

    plot_heatmap_by_criteria(filename, 4, "rank_difference", f"{base_folder}/rd_criteria_4.png")
    plot_heatmap_by_criteria(filename, 6, "rank_difference", f"{base_folder}/rd_criteria_6.png")
    plot_heatmap_by_criteria(filename, 8, "rank_difference", f"{base_folder}/rd_criteria_8.png")

    # Time plots
    plot_times_by_threshold(filename, 0.05, f"{base_folder}/time_005.png")
    plot_times_by_threshold(filename, 0.15, f"{base_folder}/time_015.png")
    plot_times_by_threshold(filename, 0.25, f"{base_folder}/time_025.png")


def complete_plots(filename: str, base_folder: str):
    # Thresholds plots
    plot_heatmap_by_threshold(filename, 0.05, "kendall_tau", f"{base_folder}/nrd_005.png")
    plot_heatmap_by_threshold(filename, 0.15, "kendall_tau", f"{base_folder}/nrd_015.png")
    plot_heatmap_by_threshold(filename, 0.25, "kendall_tau", f"{base_folder}/nrd_025.png")

    plot_heatmap_by_threshold(filename, 0.05, "normalized_hit_ratio", f"{base_folder}/nhr_005.png")
    plot_heatmap_by_threshold(filename, 0.15, "normalized_hit_ratio", f"{base_folder}/nhr_015.png")
    plot_heatmap_by_threshold(filename, 0.25, "normalized_hit_ratio", f"{base_folder}/nhr_025.png")

    plot_heatmap_by_threshold(filename, 0.05, "rank_difference", f"{base_folder}/rd_005.png")
    plot_heatmap_by_threshold(filename, 0.15, "rank_difference", f"{base_folder}/rd_015.png")
    plot_heatmap_by_threshold(filename, 0.25, "rank_difference", f"{base_folder}/rd_025.png")

    # Criteria plots
    plot_heatmap_by_criteria(filename, 4, "kendall_tau", f"{base_folder}/nrd_criteria_4.png")
    plot_heatmap_by_criteria(filename, 6, "kendall_tau", f"{base_folder}/nrd_criteria_6.png")
    plot_heatmap_by_criteria(filename, 8, "kendall_tau", f"{base_folder}/nrd_criteria_8.png")

    plot_heatmap_by_criteria(filename, 4, "normalized_hit_ratio", f"{base_folder}/nhr_criteria_4.png")
    plot_heatmap_by_criteria(filename, 6, "normalized_hit_ratio", f"{base_folder}/nhr_criteria_6.png")
    plot_heatmap_by_criteria(filename, 8, "normalized_hit_ratio", f"{base_folder}/nhr_criteria_8.png")

    plot_heatmap_by_criteria(filename, 4, "rank_difference", f"{base_folder}/rd_criteria_4.png")
    plot_heatmap_by_criteria(filename, 6, "rank_difference", f"{base_folder}/rd_criteria_6.png")
    plot_heatmap_by_criteria(filename, 8, "rank_difference", f"{base_folder}/rd_criteria_8.png")

    # Time plots
    plot_times_by_threshold(filename, 0.05, f"{base_folder}/time_005.png")
    plot_times_by_threshold(filename, 0.15, f"{base_folder}/time_015.png")
    plot_times_by_threshold(filename, 0.25, f"{base_folder}/time_025.png")

def electre3_partial_plot():
    filename = "experiments/data/electreIII/partial.json"
    base_folder = "experiments/visual/electreIII/partial"

    partial_plots(filename, base_folder)

def electre3_complete_plot():
    filename = "experiments/data/electreIII/complete.json"
    base_folder = "experiments/visual/electreIII/complete"

    complete_plots(filename, base_folder)

def promethee_partial_plot():
    filename = "experiments/data/prometheeI/partial.json"
    base_folder = "experiments/visual/promethee/partial"

    partial_plots(filename, base_folder)

def promethee_complete_plot():
    filename = "experiments/data/prometheeII/complete.json"
    base_folder = "experiments/visual/promethee/complete"

    complete_plots(filename, base_folder)

def promethee_complete_plot():
    filename = "experiments/data/prometheeII/complete.json"
    base_folder = "experiments/visual/promethee/complete"

    complete_plots(filename, base_folder)

def outranking_complete_plot():
    filename = "experiments/data/netflow/completev2.json"
    base_folder = "experiments/visual/outranking/complete"

    complete_plots(filename, base_folder)

if __name__ == "__main__":
    # plot_crisp_heatmap_by_threshold("experiments/data/netflow/completev2.json", 0.05, "rank_difference", "experiments/visual/outranking/complete/rd_005.png")
    plot_crisp_heatmap_by_criteria("experiments/data/netflow/completev2.json", 4, "rank_difference", "experiments/visual/outranking/complete/rd_criteria_4.png")
    # electre3_complete_plot()
    # promethee_partial_plot()
    # promethee_complete_plot()
    # outranking_complete_plot()

    # plot_heatmap_by_criteria("experiments/data/electreIII/partial.json", 3, "normalized_rank_difference", "experiments/visual/electreIII/partial/nrd_criteria_3.png")

    # filename="rdm.csv"
    # criteria_numbers = [3, 4, 5, 6, 7, 8]
    # alternatives_numbers = [6, 8, 10, 12, 14, 16, 18, 20]

    # data = np.loadtxt(filename, delimiter=",").reshape(len(criteria_numbers), len(alternatives_numbers))
    # print(data)
    # plot_heatmap(data, alternatives_numbers, criteria_numbers, "Number of alternatives", "Number of criteria", "Rank Difference Measure for Promethee I comparison")
    
    # plot_heatmap_by_threshold("experiments/data/prometheeI/partial.json", 0.05, "normalized_hit_ratio")
    # plot_times_by_threshold("experiments/data/prometheeI/partial.json", 0.05)

    # ===
    
    # filename = "experiments/data/electreIII/partial.json"
    # with open(filename, "r") as f:
    #     data = json.load(f)
    #     # data = np.loadtxt(f, delimiter=",")
    #     # print(data)
    #     th_low = filter(lambda x: x["thresholds"]["indifference"] == 0.05, data)
    #     th_mid = filter(lambda x: x["thresholds"]["indifference"] == 0.15, data)
    #     th_high = filter(lambda x: x["thresholds"]["indifference"] == 0.25, data)

    #     nrd_low = get_avg(th_high, "normalized_rank_difference")
    #     nrd_low.append({"alternatives": 6, "criteria": 3, "measure": 0.26})
    #     nrd_low.append({"alternatives": 6, "criteria": 4, "measure": 0.26})
    #     nrd_low.append({"alternatives": 6, "criteria": 5, "measure": 0.26})
    #     nrd_low.append({"alternatives": 6, "criteria": 6, "measure": 0.27})
    #     nrd_low.append({"alternatives": 6, "criteria": 7, "measure": 0.24})
    #     nrd_low.append({"alternatives": 6, "criteria": 8, "measure": 0.28})


    #     df = pd.DataFrame(nrd_low)

    #     # print(df)
    #     # Przekształcenie danych na dwuwymiarową tablicę
    #     pivot_df = df.pivot(index='criteria', columns='alternatives', values='measure')
    #     print(pivot_df)
    #     # nrd_matrix = as_matrix(nrd_low, [6, 8, 10, 12, 14, 16, 18, 20], [3, 4, 5, 6, 7, 8])
    #     # print(nrd_matrix)
        
    #     # print(kendall_low)
    #     plot_heatmap(pivot_df, [6, 8, 10, 12, 14, 16, 18, 20], [3, 4, 5, 6, 7, 8], "Number of alternatives", "Number of criteria", "")


