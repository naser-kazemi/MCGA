from pymoo.problems import get_problem
from emoa.moop import MOOP

import matplotlib.pyplot as plt
from emoa.utils import *
from nsga2.model import NSGA2
from nsga3.model import NSGA3
import math

from deap import benchmarks


def run():
    problem_name = "dtlz2"
    problem = get_problem(problem_name)
    lower_bound = 0.0
    upper_bound = 1.0

    population_size = 150
    num_variables = problem.n_var
    num_objectives = problem.n_obj
    num_generations = 250
    eta_crossover = 20
    eta_mutation = 20
    crossover_probability = 0.6
    num_divisions = 5

    model = NSGA3(
        problem=lambda ind: benchmarks.dtlz2(ind, 3),
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
    ax.scatter(
        problem.pareto_front()[:, 0],
        problem.pareto_front()[:, 1],
        problem.pareto_front()[:, 2],
        color="red",
        alpha=0.5,
        label="Pareto Front",
    )

    ax.set_xlabel("$f_1()$", fontsize=15)
    ax.set_ylabel("$f_2()$", fontsize=15)
    ax.set_zlabel("$f_3()$", fontsize=15)
    ax.view_init(30, 40)
    plt.autoscale(tight=True)
    plt.savefig("images/dtlz2_nsga3.png", dpi=300)

    hypervolumes = model.hypervolume(log=True)
    fig = plt.figure(figsize=(7, 7))
    plt.plot(hypervolumes)
    plt.xlabel("Iterations (t)")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume over time")
    plt.savefig("images/dtlz2_nsga3_hypervolume.png", dpi=300)


if __name__ == "__main__":
    run()
