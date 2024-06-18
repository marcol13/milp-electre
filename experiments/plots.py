import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(matrix: np.array, xticks_label: list[str], yticks_label: list[str], xlabel: str, ylabel: str, title: str):
    fig, ax = plt.subplots()

    cax = ax.matshow(matrix, cmap='viridis')
    fig.colorbar(cax)

    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_xticklabels([''] + xticks_label)
    ax.set_yticklabels([''] + yticks_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_title(title)

    plt.show()


if __name__ == "__main__":
    filename="rdm.csv"
    criteria_numbers = [3, 4, 5, 6, 7, 8]
    alternatives_numbers = [6, 8, 10, 12, 14, 16, 18, 20]

    data = np.loadtxt(filename, delimiter=",").reshape(len(criteria_numbers), len(alternatives_numbers))
    print(data)
    plot_heatmap(data, alternatives_numbers, criteria_numbers, "Number of alternatives", "Number of criteria", "Rank Difference Measure for Promethee I comparison")