import json

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os
import sys


def plot_hypervolume_util(test_suit, test, model_name, ax, color):
    path = f"result/{test_suit}/{test}/{model_name}/"
    # get all hypervolume json files in the directory
    files = [f for f in os.listdir(path) if "hypervolume" in f]
    hypervolumes = []
    for file in files:
        with open(path + file, "r") as f:
            hypervolumes.append(json.load(f))

    mean_hypervolumes = np.mean(hypervolumes, axis=0)
    std_hypervolumes = np.std(hypervolumes, axis=0)
    # print(std_hypervolumes)

    # plot the mean hypervolume with std as error bar
    x = np.arange(len(mean_hypervolumes))
    ax.plot(x, mean_hypervolumes, label=model_name, color=color)
    ax.fill_between(
        x,
        mean_hypervolumes - std_hypervolumes,
        mean_hypervolumes + std_hypervolumes,
        alpha=0.4,
        color=color,
    )


def plot_hypevolumes(test_suite, test, model_names):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    colors = [
        "blue",
        "red",
        "orange",
        "green",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    for i, model_name in enumerate(model_names):
        plot_hypervolume_util(test_suite, test, model_name, ax, colors[i])

    ax.set_xlabel("Iterations (t)")
    ax.set_ylabel("Hypervolume")
    ax.set_title("Hypervolume over time")
    ax.legend()
    plt.show()


def get_mean_population(test_suit, test, model_name):
    path = f"result/{test_suit}/{test}/{model_name}/"
    # get all hypervolume json files in the directory
    files = [f for f in os.listdir(path) if "population" in f]
    populations = []
    for file in files:
        with open(path + file, "r") as f:
            pop = json.load(f)
            pop = np.array(pop)
            populations.append(pop)

    # return np.mean(populations, axis=0)
    return populations[0]


def plot_populations(test_suite, test, model_name, num_objectives):
    fig = plt.figure(figsize=(7, 7))
    if num_objectives == 3:
        ax = fig.add_subplot(111, projection="3d")
        mean_population = get_mean_population(test_suite, test, model_name)
        ax.scatter(
            mean_population[:, 0],
            mean_population[:, 1],
            mean_population[:, 2],
            color="blue",
            alpha=0.5,
            label=model_name,
        )
        ax.set_zlabel("$f_3$", fontsize=15)
    else:
        ax = fig.add_subplot(111)
        mean_population = get_mean_population(test_suite, test, model_name)
        ax.scatter(
            mean_population[:, 0],
            mean_population[:, 1],
            color="blue",
            alpha=0.5,
            label=model_name,
        )

    ax.set_xlabel("$f_1$", fontsize=15)
    ax.set_ylabel("$f_2$", fontsize=15)
    ax.set_title("Population")
    ax.legend()
    plt.show()


def main():
    test_suite = "re"
    test = "re5"
    num_objectives = 3
    model_names = ["nsga3", "mc_nsga3"]
    sns.set_theme(style="darkgrid")
    plot_hypevolumes(test_suite, test, model_names)
    plot_populations(test_suite, test, model_names[0], num_objectives)
    plot_populations(test_suite, test, model_names[1], num_objectives)


if __name__ == "__main__":
    main()
