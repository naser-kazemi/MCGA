from pymoo.problems import get_problem
from emoa.moop import MOOP

import matplotlib.pyplot as plt
from emoa.utils import *
from mcga import MCNSGA3
from nsga2 import NSGA2
from mcga import MCNSGA2
import math
import json

from deap import benchmarks, tools

from nsga3 import NSGA3

problem_name = "dtlz4"
pymoo_problem = get_problem(problem_name)
lower_bound = 0.0
upper_bound = 1.0
population_size = 300
num_variables = pymoo_problem.n_var
num_objectives = 5
num_generations = 200
num_divisions = 8
eta_crossover = 20
eta_mutation = 20
crossover_probability = 0.8
polar_offset_limit = (0, np.pi / 2)
num_max_sectors = (3, 7)
front_frequency_threshold = 0.1
monte_carlo_frequency = 5
log = ["hv"]
verbose = True
expr = 1
model_name = ""
problem = lambda ind: benchmarks.dtlz3(ind, 3)


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


def nsga3_model():
    global model_name
    model_name = "nsga3"
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
    global model_name
    model_name = "mc_nsga3"
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
        monte_carlo_frequency=monte_carlo_frequency,
        log=log,
        verbose=verbose,
    )


def run_nsga(selected_model=None):
    if selected_model is not None:
        if selected_model == "nsga3":
            model = nsga3_model()
        elif selected_model == "mc_nsga3":
            model = mcnsga3_model()
        elif selected_model == "nsga2":
            model = nsga2_model()
        else:
            model = mcnsga2_model()
    else:
        # model = nsga2_model()
        # model = mcnsga2_model()
        # model = nsga3_model()
        model = mcnsga3_model()

    model.run()

    individuals = model.result_pop
    individuals = np.array([ind.fitness.values for ind in individuals])

    sns.set_theme(style="darkgrid")
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # plot true pareto front
    ax.scatter(
        pymoo_problem.pareto_front()[:, 0],
        pymoo_problem.pareto_front()[:, 1],
        pymoo_problem.pareto_front()[:, 2],
        color="red",
        alpha=0.5,
        label="Optimal Pareto Front",
    )

    ax.scatter(
        individuals[:, 0],
        individuals[:, 1],
        individuals[:, 2],
        color="blue",
        alpha=0.5,
        label=model_name,
    )

    ax.set_xlabel("$f_1$", fontsize=15)
    ax.set_ylabel("$f_2$", fontsize=15)
    ax.set_zlabel("$f_3$", fontsize=15)

    # hv_ref should be the utopia point
    hv_ref_point = np.ones(num_objectives) * 100
    # hv_ref_point = np.array([5, 5, 5])
    # hv_ref_point = np.array([2, 2, 2])

    # plt.savefig("images/dtlz1_nsga3.png", dpi=300)
    # plt.show()

    hypervolumes = model.metric("hypervolume", all_gens=True, ref=hv_ref_point)
    fig = plt.figure(figsize=(7, 7))
    plt.plot(hypervolumes)
    plt.xlabel("Iterations (t)")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume over time")
    # plt.savefig("images/dtlz1_nsga3_hypervolume.png", dpi=300)
    # plt.show()

    path = f"result/dtlz5d/{problem_name}/{model_name}"

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

    # history = np.array(model.logbook.select("pop"))
    # make_gif_from_history(history, path + f"/generations{expr}.gif")


def run():
    # problem_names = ["dtlz2", "dtlz3", "dtlz4"]
    problem_names = ["dtlz1"]
    problems = [
        lambda ind: benchmarks.dtlz1(ind, 5),
        # lambda ind: benchmarks.dtlz2(ind, 5),
        # lambda ind: benchmarks.dtlz3(ind, 5),
        # lambda ind: benchmarks.dtlz4(ind, 5, 100),
    ]
    global problem_name
    global problem
    global expr
    for (pn, p) in zip(problem_names, problems):
        problem_name = pn
        problem = p
        for i in range(1, 5 + 1):
            expr = i
            # run_nsga("nsga2")
            run_nsga("mc_nsga2")
            # run_nsga("nsga3")
            # run_nsga("mc_nsga3")


if __name__ == "__main__":
    run()
