import array
import copy

import numpy as np
import json

from deap.tools._hypervolume import hv

from nsga2 import NSGA2
from nsga3 import NSGA3
from mcga import MCNSGA3

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


def gamma6(xi):
    k = 2
    return np.power(np.abs(2 * (k * xi[2] - np.floor(k * xi[2])) - 1), 0.5 + xi[0])


def cpft6(xi):
    k = 2
    a = np.array([xi[0], xi[0], -((2 * xi[0] - 1) ** 3)])
    b = np.array([2 * k * xi[1], gamma6(xi), 0])
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
num_divisions = 12
polar_offset_limit = np.pi
num_max_sectors = 30
front_frequency_threshold = 0.01
niche_ratio = 0.05
monte_carlo_frequency = 4
verbose = True
log = ["hv"]
problem = cpft3
problem_name = "cpft3"
model_name = ""
expr = 1


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
        niche_ratio=niche_ratio,
        monte_carlo_frequency=monte_carlo_frequency,
        log=log,
        verbose=verbose,
    )


def run_nsga(selected_model=None):
    if selected_model is not None:
        if selected_model == "nsga3":
            model = nsga3_model()
        else:
            model = mcnsga3_model()
    else:
        model = nsga3_model()
        # model = mcnsga3_model()

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
        label=model_name,
    )

    ax.set_xlabel("$f_1$", fontsize=15)
    ax.set_ylabel("$f_2$", fontsize=15)
    ax.set_zlabel("$f_3$", fontsize=15)
    # ax.view_init(30, 40)
    # plt.autoscale(tight=True)
    # plt.savefig(f"images/{problem_name}_mc_nsga3.png", dpi=300)
    # plt.savefig(f"images/{problem_name}_nsga3.png", dpi=300)
    plt.show()

    # hv_ref_point = np.array([-0.5, -0.5, -0.5])
    hv_ref_point = np.array([1, 1, 1])

    hypervolumes = model.metric("hypervolume", all_gens=True, ref=hv_ref_point)
    fig = plt.figure(figsize=(7, 7))
    plt.plot(hypervolumes)
    plt.xlabel("Iterations (t)")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume over time")
    # plt.savefig(f"images/{problem_name}_mc_nsga3_hypervolume.png", dpi=300)
    # plt.savefig(f"images/{problem_name}_nsga3_hypervolume.png", dpi=300)
    plt.show()

    path = f"result/cpft/{problem_name}/{model_name}"

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
        if model_name == "mc_nsga3":
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


def run():
    problem_names = ["cpft2", "cpft3", "cpft4"]
    problems = [cpft2, cpft3, cpft3]
    global problem_name
    global problem
    global expr
    for (pn, p) in zip(problem_names, problems):
        problem_name = pn
        problem = p
        for i in range(1, 5 + 1):
            expr = i
            run_nsga("nsga3")
            run_nsga("mc_nsga3")


if __name__ == "__main__":
    run()
    # main()

# def main():
#     # test the deap implementation of NSGA3
#     from deap import base, creator, tools, algorithms
#     from deap.tools import selNSGA3
#
#     creator.create(
#         "FitnessMin",
#         base.Fitness,
#         weights=(-1.0,) * 3,
#         crowding_dist=0.0,
#     )
#     creator.create(
#         "Individual", array.array, typecode="d", fitness=creator.FitnessMin
#     )
#
#     toolbox = base.Toolbox()
#     toolbox.register("attr_float", uniform, lower_bound, upper_bound, num_variables)
#     toolbox.register(
#         "individual", tools.initIterate, creator.Individual, toolbox.attr_float
#     )
#     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#     toolbox.register("evaluate", cpft4)
#     toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=lower_bound, up=upper_bound, eta=eta_crossover)
#     toolbox.register("mutate", tools.mutPolynomialBounded, low=lower_bound, up=upper_bound, eta=eta_mutation,
#                      indpb=1.0 / num_variables)
#     toolbox.register("select", selNSGA3, ref_points=tools.uniform_reference_points(nobj=3, p=12))
#
#     # run the algorithm
#     pop = toolbox.population(n=population_size)
#
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("pop", copy.deepcopy)
#
#     res, logbook = algorithms.eaMuPlusLambda(
#         pop,
#         toolbox,
#         mu=population_size,
#         lambda_=population_size,
#         cxpb=crossover_probability,
#         mutpb=1.0 / num_variables,
#         ngen=300,
#         stats=stats,
#         verbose=False,
#     )
#
#     # plot the final population and hypervolume
#     individuals = np.array([ind.fitness.values for ind in res])
#     fig = plt.figure(figsize=(7, 7))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.scatter(
#         individuals[:, 0],
#         individuals[:, 1],
#         individuals[:, 2],
#         color="blue",
#         alpha=0.5,
#         label=model_name,
#     )
#
#     ax.set_xlabel("$f_1$", fontsize=15)
#     ax.set_ylabel("$f_2$", fontsize=15)
#     ax.set_zlabel("$f_3$", fontsize=15)
#     plt.show()
#
#     hv_ref_point = np.array([1, 1, 1])
#
#     def hyper_volume_util(population, ref=None):
#         # front = self.nd_sort(population, len(population), first_front_only=True)
#         # objs = np.array([ind.fitness.values for ind in population]) * -1
#         # print(population)
#         # objs = np.array([ind.fitness.wvalues for ind in front]) * -1
#         objs = np.array(population)
#         if ref is None:
#             ref = np.max(objs, axis=0) + 1
#         return hv.hypervolume(objs, ref)
#
#     hypervolumes = [hyper_volume_util(logbook.select("pop")[i], hv_ref_point) for i in
#                     range(len(logbook.select("pop")))]
#
#     fig = plt.figure(figsize=(7, 7))
#     plt.plot(hypervolumes)
#     plt.xlabel("Iterations (t)")
#     plt.ylabel("Hypervolume")
#     plt.title("Hypervolume over time")
#     plt.show()
