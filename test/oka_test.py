import matplotlib.pyplot as plt
import numpy as np
import json

from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

from nsga2 import NSGA2
from nsga3 import NSGA3
from mcga import MCNSGA2
from mcga import MCNSGA3
from moead import MOEAD

from emoa.utils import *


class OKA1(Problem):
    def __init__(self):
        sin, cos = np.sin(np.pi / 12), np.cos(np.pi / 12)
        super().__init__(
            n_var=2,
            n_obj=2,
            n_constr=0,
            xl=[6 * sin, -2 * np.pi * sin],
            xu=[6 * sin + 2 * np.pi * cos, 6 * cos],
            type_var=np.double,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        sin, cos = np.sin(np.pi / 12), np.cos(np.pi / 12)
        x1, x2 = x[:, 0], x[:, 1]
        x1_ = cos * x1 - sin * x2
        x2_ = sin * x1 + cos * x2

        f1 = x1_
        f2 = np.sqrt(2 * np.pi) - np.sqrt(np.abs(x1_)) + 2 * np.abs(x2_ - 3 * np.cos(x1_) - 3) ** (1. / 3)

        return np.column_stack([f1, f2])

    def _calc_pareto_front(self, n_pareto_points=100):
        x1_ = np.linspace(0, 2 * np.pi, n_pareto_points)
        x2_ = 3 * np.cos(x1_) + 3
        x = np.array([x1_, x2_]).T
        return self._evaluate(x, out=None)


def oka1(x):
    sin, cos = np.sin(np.pi / 12), np.cos(np.pi / 12)
    x1, x2 = x[0], x[1]
    x1_ = cos * x1 - sin * x2
    x2_ = sin * x1 + cos * x2

    f1 = x1_
    f2 = np.sqrt(2 * np.pi) - np.sqrt(np.abs(x1_)) + 2 * np.abs(x2_ - 3 * np.cos(x1_) - 3) ** (1. / 3)

    return np.array([f1, f2])


class OKA2(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, type_var=np.double)
        self.xl = np.array([-np.pi, -5.0, -5.0])
        self.xu = np.array([np.pi, 5.0, 5.0])

    def _evaluate_F(self, x):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]

        f1 = x1
        f2 = 1 - (x1 + np.pi) ** 2 / (4 * np.pi ** 2) + \
             np.abs(x2 - 5 * np.cos(x1)) ** (1. / 3) + np.abs(x3 - 5 * np.sin(x1)) ** (1. / 3)

        return np.column_stack([f1, f2])

    def _calc_pareto_front(self, n_pareto_points=100):
        f1 = np.linspace(-np.pi, np.pi, n_pareto_points)
        f2 = 1 - (f1 + np.pi) ** 2 / (4 * np.pi ** 2)
        return np.column_stack([f1, f2])


def oka2(x):
    x1, x2, x3 = x[0], x[1], x[2]

    f1 = x1
    f2 = 1 - (x1 + np.pi) ** 2 / (4 * np.pi ** 2) + \
         np.abs(x2 - 5 * np.cos(x1)) ** (1. / 3) + np.abs(x3 - 5 * np.sin(x1)) ** (1. / 3)

    return np.array([f1, f2])


population_size = 500
num_variables = 3
num_objectives = 2
num_generations = 750
eta_crossover = 20
eta_mutation = 20
crossover_probability = 0.75
sin, cos = np.sin(np.pi / 12), np.cos(np.pi / 12)
# lower_bound = [6 * sin, -2 * np.pi * sin]  # OKA1
# upper_bound = [6 * sin + 2 * np.pi * cos, 6 * cos]  # OKA1
lower_bound = [-np.pi, -5.0, -5.0]  # OKA2
upper_bound = [np.pi, 5.0, 5.0]  # OKA2
num_divisions = 8
polar_offset_limit = np.pi
num_max_sectors = 30
front_frequency_threshold = 0.1
niche_ratio = 0.15
monte_carlo_frequency = 2
verbose = True
log = ["hv"]
problem = oka2
problem_name = "oka2"
pymoo_problem = OKA2()
model_name = ""
expr = 1


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
        niche_ratio=niche_ratio,
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

    # plot using seaborn
    sns.set_theme(style="darkgrid")
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    # ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pymoo_problem.pareto_front()[:, 0],
        pymoo_problem.pareto_front()[:, 1],
        color="red",
        alpha=0.3,
        label="Optimal Pareto Front",
    )

    ax.scatter(
        individuals[:, 0],
        individuals[:, 1],
        # individuals[:, 2],
        color="blue",
        alpha=0.6,
        label=model_name,
    )

    ax.set_xlabel("$f_1$", fontsize=15)
    ax.set_ylabel("$f_2$", fontsize=15)
    # ax.set_zlabel("$f_3$", fontsize=15)
    # ax.view_init(30, 40)
    # plt.autoscale(tight=True)
    # plt.savefig(f"images/{problem_name}_mc_nsga3.png", dpi=300)
    # plt.savefig(f"images/{problem_name}_nsga3.png", dpi=300)
    # plt.show()

    # hv_ref_point = np.array([-0.5, -0.5, -0.5])
    hv_ref_point = np.max(OKA1().pareto_front(), axis=0) + 10
    # hv_ref_point = np.array([1000, 1000])

    hypervolumes = model.metric("hypervolume", all_gens=True, ref=hv_ref_point)
    fig = plt.figure(figsize=(7, 7))
    plt.plot(hypervolumes)
    plt.xlabel("Iterations (t)")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume over time")
    # plt.savefig(f"images/{problem_name}_mc_nsga3_hypervolume.png", dpi=300)
    # plt.savefig(f"images/{problem_name}_nsga3_hypervolume.png", dpi=300)
    # plt.show()

    path = f"result/oka/{problem_name}/{model_name}"

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

    # history = np.array(model.logbook.select("pop"))
    # make_gif_from_history(history, path + f"/generations{expr}.gif")


def run_moead():
    problem = pymoo_problem
    pop_size = 500
    ref_dirs = get_reference_directions("das-dennis", num_objectives, n_partitions=20)
    n_neighbors = 20
    decomposition = None
    prob_neighbor_mating = 0.9
    sampling = FloatRandomSampling()
    cross_prob = 0.9
    crossover = SBX(prob=cross_prob, eta=eta_crossover)
    mutation = PM(prob_var=None, eta=eta_mutation)
    n_gen = 300
    hv_ref_point = np.max(problem.pareto_front(), axis=0) + 10

    global model_name
    model_name = "moead"
    model = MOEAD(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        n_neighbors=n_neighbors,
        decomposition=decomposition,
        prob_neighbor_mating=prob_neighbor_mating,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        hv_ref=hv_ref_point,
    )

    res = minimize(
        problem,
        model,
        termination=("n_gen", n_gen),
        seed=1,
        verbose=True,
        save_history=True,
    )

    # Scatter().add(res.history[0].pop.get("F")).show()
    # plt.show()

    hypervolumes = MOEAD.history_hypervolume(res.history, ref=hv_ref_point)
    # print(hypervolumes)
    fig = plt.figure(figsize=(7, 7))
    plt.plot(hypervolumes)
    plt.xlabel("Iterations (t)")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume over time")
    # plt.savefig(f"images/{problem_name}_mc_nsga3_hypervolume.png", dpi=300)
    # plt.savefig(f"images/{problem_name}_nsga3_hypervolume.png", dpi=300)
    # plt.show()

    path = f"result/oka/{problem_name}/{model_name}"

    with open(path + "/experiment.txt", "w") as f:
        f.write(f"problem: {problem_name}\n")
        f.write(f"population_size: {len(res.F)}\n")
        f.write(f"num_variables: {problem.n_var}\n")
        f.write(f"num_objectives: {problem.n_obj}\n")
        f.write(f"num_generations: {n_gen}\n")
        f.write(f"eta_crossover: {eta_crossover}\n")
        f.write(f"eta_mutation: {eta_mutation}\n")
        f.write(f"crossover_probability: {cross_prob}\n")
        f.write(f"lower_bound: {lower_bound}\n")
        f.write(f"upper_bound: {upper_bound}\n")
        f.write(f"hv_ref_point: {hv_ref_point}\n")
        if model_name == "mc_moead":
            f.write(f"polar_offset_limit: {polar_offset_limit}\n")
            f.write(f"num_max_sectors: {num_max_sectors}\n")
            f.write(f"front_frequency_threshold: {front_frequency_threshold}\n")
            f.write(f"monte_carlo_frequency: {monte_carlo_frequency}\n")

    # save the final population
    with open(path + f"/population{expr}.json", "w") as f:
        json.dump(res.F.tolist(), f)

    # save hypervolume data
    with open(path + f"/hypervolume{expr}.json", "w") as f:
        json.dump(hypervolumes, f)


def run():
    # run_nsga("nsga3")
    # run_nsga("mc_nsga3")

    problem_names = ["oka1", "oka2"]
    pymoo_problems = [OKA1(), OKA2()]
    problems = [oka1, oka2]
    global problem_name
    global problem
    global expr
    global pymoo_problem
    global num_objectives
    global num_variables
    for (pn, p, pymoop) in zip(problem_names, problems, pymoo_problems):
        problem_name = pn
        problem = p
        pymoo_problem = pymoop
        num_objectives = pymoo_problem.n_obj
        num_variables = pymoo_problem.n_var
        for i in range(1, 5 + 1):
            expr = i
            run_nsga("nsga3")
            run_nsga("mc_nsga3")
    # run_nsga("nsga3")
    # run_nsga("mc_nsga3")


def plot_oka_pareto_front(ax, test):
    if test == "oka1":
        problem = OKA1()
    else:
        problem = OKA2()

    ax.scatter(
        problem.pareto_front()[:, 0],
        problem.pareto_front()[:, 1],
        color="red",
        alpha=0.3,
        label="Optimal Pareto Front",
    )


if __name__ == "__main__":
    run()
