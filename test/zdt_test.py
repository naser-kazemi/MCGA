from pymoo.problems import get_problem
from emoa.moop import MOOP

import matplotlib.pyplot as plt
from emoa.utils import *
from nsga2.model import NSGA2
from mcga.mc_nsga2 import MCNSGA2
import math
import json

from deap import benchmarks, tools

problem_name = "zdt4"
pymoo_problem = get_problem(problem_name)
lower_bound = 0.0
upper_bound = 1.0
population_size = 100
num_variables = pymoo_problem.n_var
num_objectives = pymoo_problem.n_obj
num_generations = 2000
eta_crossover = 20
eta_mutation = 20
crossover_probability = 0.8
polar_offset_limit = np.pi
num_max_sectors = 30
front_frequency_threshold = 0.01
monte_carlo_frequency = 2
log = ["hv"]
verbose = True
expr = 5
model_name = ""
problem = lambda ind: benchmarks.zdt4(ind)


def nsga2_model():
    global model_name
    model_name = "nsga2"
    return NSGA2(
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
        log=log,
        verbose=verbose,
    )


def mcnsga2_model():
    global model_name
    model_name = "mc_nsga2"
    return MCNSGA2(
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
        polar_offset_limit=polar_offset_limit,
        num_max_sectors=num_max_sectors,
        front_frequency_threshold=front_frequency_threshold,
        monte_carlo_frequency=monte_carlo_frequency,
        log=log,
        verbose=verbose,
    )


def run():
    # model = nsga2_model()
    model = mcnsga2_model()

    model.run()

    fig = plt.figure(figsize=(7, 7))
    sns.set_theme(style="darkgrid")

    individuals = model.result_pop
    individuals = np.array([ind.fitness.values for ind in individuals])

    # plot true pareto front
    plt.scatter(
        pymoo_problem.pareto_front()[:, 0],
        pymoo_problem.pareto_front()[:, 1],
        color="red",
        alpha=0.5,
        label="Optimal Pareto Front",
    )

    plt.scatter(
        individuals[:, 0],
        individuals[:, 1],
        color="blue",
        alpha=0.5,
        label="Pareto Front",
    )

    # hv_ref should be the utopia point
    hv_ref_point = np.max(pymoo_problem.pareto_front(), axis=0) + 1

    plt.xlabel("$f_1(x)$", fontsize=15)
    plt.ylabel("$f_2(x)$", fontsize=15)
    # plt.savefig("images/dtlz1_nsga3.png", dpi=300)
    plt.show()

    hypervolumes = model.metric("hypervolume", all_gens=True, ref=hv_ref_point)
    fig = plt.figure(figsize=(7, 7))
    plt.plot(hypervolumes)
    plt.xlabel("Iterations (t)")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume over time")
    # plt.savefig("images/dtlz1_nsga3_hypervolume.png", dpi=300)
    plt.show()

    path = f"result/zdt/{problem_name}/{model_name}"

    with open(path + "/experiment.txt", "w") as f:
        f.write(f"problem: {problem_name}\n")
        f.write(f"population_size: {population_size}\n")
        f.write(f"num_variables: {num_variables}\n")
        f.write(f"num_objectives: {num_objectives}\n")
        f.write(f"num_generations: {num_generations}\n")
        f.write(f"eta_crossover: {eta_crossover}\n")
        f.write(f"eta_mutation: {eta_mutation}\n")
        f.write(f"crossover_probability: {crossover_probability}\n")
        f.write(f"lower_bound: {lower_bound}\n")
        f.write(f"upper_bound: {upper_bound}\n")
        f.write(f"hv_ref_point: {hv_ref_point}\n")
        if model_name == "mc_nsga2":
            f.write(f"polar_offset_limit: {polar_offset_limit}\n")
            f.write(f"num_max_sectors: {num_max_sectors}\n")
            f.write(f"front_frequency_threshold: {front_frequency_threshold}\n")
            f.write(f"monte_carlo_frequency: {monte_carlo_frequency}\n")

    # save the final population
    with open(path + f"/population{expr}.json", "w") as f:
        json.dump(individuals.tolist(), f)

    # save hypervolume data
    with open(path + f"/hypervolume{expr}.json", "w") as f:
        json.dump(hypervolumes, f)

    if __name__ == "__main__":
        run()
