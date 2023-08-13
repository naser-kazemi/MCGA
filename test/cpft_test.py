import numpy as np

from nsga2 import NSGA2
from nsga3 import NSGA3

from pymoo.problems import get_problem
from emoa.moop import MOOP
from emoa.utils import *

A = np.array(
    [
        [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
        [-np.cos(np.pi / 4), np.sin(np.pi / 4), 0],
        [0, 0, 0],
    ]
)


def gamma(xi):
    return np.power(np.abs(2 * xi[1] - 1), xi[0] + 0.5)


def cpft3(ind):
    xi = ind[:]
    a = np.array([xi[0], xi[1], -(2 * xi[0] - 1) ** 3])
    b = np.array([2 * xi[1] - 1, gamma(xi), 0])
    return a + A @ b


def run():
    population_size = 500
    num_variables = 7
    num_objectives = 3
    num_generations = 700
    eta_crossover = 20
    eta_mutation = 20
    crossover_probability = 0.6
    lower_bound = 0.0
    upper_bound = 1.0

    model = NSGA3(
        problem=cpft3,
        population_size=population_size,
        num_variables=num_variables,
        num_objectives=num_objectives,
        num_generations=num_generations,
        eta_crossover=eta_crossover,
        eta_mutation=eta_mutation,
        crossover_probability=crossover_probability,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        num_divisions=5,
    )

    model.run(verbose=False)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    for ind in model.result_pop:
        ax.scatter(
            ind.fitness.values[0],
            ind.fitness.values[1],
            ind.fitness.values[2],
            marker="o",
            color="blue",
        )

    # plot true pareto front
    # ax.scatter(
    #     problem.pareto_front()[:, 0],
    #     problem.pareto_front()[:, 1],
    #     problem.pareto_front()[:, 2],
    #     color="red",
    #     alpha=0.5,
    #     label="Pareto Front",
    # )

    ax.set_xlabel("$f_1()$", fontsize=15)
    ax.set_ylabel("$f_2()$", fontsize=15)
    ax.set_zlabel("$f_3()$", fontsize=15)
    # ax.view_init(30, 40)
    plt.autoscale(tight=True)
    plt.savefig("images/cpft3_nsga3.png", dpi=300)

    hypervolumes = model.hypervolume(log=True)
    fig = plt.figure(figsize=(7, 7))
    plt.plot(hypervolumes)
    plt.xlabel("Iterations (t)")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume over time")
    plt.savefig("images/cpft3_nsga3_hypervolume.png", dpi=300)


if __name__ == "__main__":
    ...