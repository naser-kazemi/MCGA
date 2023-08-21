import numpy as np

from nsga2 import NSGA2
from nsga3 import NSGA3
from mcga import MCNSGA3

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


def gamma2(xi):
    return 0.5 * (2 * xi[0] - 1) * np.power(2 * xi[1] - 1, 2)


def cpft2(ind):
    xi = ind[:]
    a = np.array([xi[0], xi[0], -((2 * xi[0] - 1) ** 3)])
    b = np.array([2 * xi[1] - 1, gamma2(xi), 0])
    return a + A @ b


def gamma3(xi):
    return np.power(np.abs(2 * xi[1] - 1), xi[0] + 0.5)


def cpft3(ind):
    xi = ind[:]
    a = np.array([xi[0], xi[0], -((2 * xi[0] - 1) ** 3)])
    b = np.array([2 * xi[1] - 1, gamma3(xi), 0])
    return a + A @ b


def gamma4(xi):
    return np.power(np.abs(2 * xi[1] - 1), 1.0 - 0.5 * np.sin(4 * np.pi * xi[0]))


def cpft4(ind):
    xi = ind[:]
    a = np.array([xi[0], xi[0], -((2 * xi[0] - 1) ** 3)])
    b = np.array([2 * xi[1] - 1, gamma4(xi), 0])
    return a + A @ b


population_size = 1000
num_variables = 7
num_objectives = 3
num_generations = 300
eta_crossover = 20
eta_mutation = 20
crossover_probability = 0.6
lower_bound = 0.0
upper_bound = 1.0
num_divisions = 8
polar_offset_limit = np.pi
num_max_sectors = 30
front_frequency_threshold = 0.01
niche_ratio = 0.05
monte_carlo_frequency = 4
verbose = True
log = ["hv"]
problem = cpft4
problem_name = "cpft4"


def nsga3_model():
    return NSGA3(
        problem=problem,
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
        log=log,
        verbose=verbose,
    )


def mcnsga3_model():
    return MCNSGA3(
        problem=problem,
        population_size=population_size,
        num_variables=num_variables,
        num_objectives=num_objectives,
        num_generations=num_generations,
        eta_crossover=eta_crossover,
        eta_mutation=eta_mutation,
        crossover_probability=crossover_probability,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        num_divisions=num_divisions,
        polar_offset_limit=polar_offset_limit,
        num_max_sectors=num_max_sectors,
        front_frequency_threshold=front_frequency_threshold,
        niche_ratio=niche_ratio,
        monte_carlo_frequency=monte_carlo_frequency,
        log=log,
        verbose=verbose,
    )


def run():
    # model = nsga3_model()
    model = mcnsga3_model()

    model.run()

    individuals = model.result_pop
    individuals = np.array([ind.fitness.values for ind in individuals])

    # plot using seaborn
    sns.set_theme(style="darkgrid")
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        individuals[:, 0],
        individuals[:, 1],
        individuals[:, 2],
        color="blue",
        alpha=0.5,
        label="NSGA3",
        # label="NSGA3",
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

    ax.set_xlabel("$f_1$", fontsize=15)
    ax.set_ylabel("$f_2$", fontsize=15)
    ax.set_zlabel("$f_3$", fontsize=15)
    # ax.view_init(30, 40)
    plt.autoscale(tight=True)
    plt.savefig(f"images/{problem_name}_mc_nsga3.png", dpi=300)
    # plt.savefig(f"images/{problem_name}_nsga3.png", dpi=300)
    plt.show()

    hypervolumes = model.metric("hypervolume", all_gens=True)
    fig = plt.figure(figsize=(7, 7))
    plt.plot(hypervolumes)
    plt.xlabel("Iterations (t)")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume over time")
    plt.savefig(f"images/{problem_name}_mc_nsga3_hypervolume.png", dpi=300)
    # plt.savefig(f"images/{problem_name}_nsga3_hypervolume.png", dpi=300)


if __name__ == "__main__":
    run()
    # main()
