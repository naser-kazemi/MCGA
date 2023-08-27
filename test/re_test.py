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


def calc_pareto_front(name):
    file_path = f'data/RE/ParetoFront/{name}.npy'
    return np.load(file_path)


def div(x1, x2):
    '''
    Divide x1 / x2, return 0 where x2 == 0
    '''
    return np.divide(x1, x2, out=np.zeros(np.broadcast(x1, x2).shape), where=(x2 != 0))


class RE1(Problem):
    def __init__(self):
        super().__init__(
            n_var=4,
            n_obj=2,
            n_constr=0,
            xl=[1, np.sqrt(2), np.sqrt(2), 1],
            xu=[3, 3, 3, 3],
            type_var=np.double,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        F = 10
        E = 2e5
        L = 200

        f1 = L * ((2 * x1) + np.sqrt(2.0) * x2 + np.sqrt(x3) + x4)
        f2 = (F * L) / E * (div(2.0, x1) + div(2.0 * np.sqrt(2.0), x2) - div(2.0 * np.sqrt(2.0), x3) + div(2.0, x4))

        out["F"] = np.column_stack([f1, f2])


def re1(x):
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]

    F = 10
    E = 2e5
    L = 200

    f1 = L * ((2 * x1) + np.sqrt(2.0) * x2 + np.sqrt(x3) + x4)
    f2 = (F * L) / E * (div(2.0, x1) + div(2.0 * np.sqrt(2.0), x2) - div(2.0 * np.sqrt(2.0), x3) + div(2.0, x4))

    return np.array([f1, f2])


class RE5(Problem):
    def __init__(self):
        super().__init__(
            n_var=4,
            n_obj=3,
            n_constr=0,
            xl=[55, 75, 1000, 11],
            xu=[80, 110, 3000, 20],
            type_var=np.double,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        f1 = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
        f2 = div((9.82 * 1e6) * (x2 * x2 - x1 * x1), x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))

        g = np.column_stack([
            (x2 - x1) - 20.0,
            0.4 - div(x3, (3.14 * (x2 * x2 - x1 * x1))),
            1.0 - div(2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1), np.power((x2 * x2 - x1 * x1), 2)),
            div(2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1), x2 * x2 - x1 * x1) - 900.0
        ])

        g[g >= 0] = 0
        g[g < 0] = -g[g < 0]

        f3 = np.sum(g, axis=1)

        out["F"] = np.column_stack([f1, f2, f3])


def re5(x):
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]

    f1 = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
    f2 = div((9.82 * 1e6) * (x2 * x2 - x1 * x1), x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))

    g = np.column_stack([
        (x2 - x1) - 20.0,
        0.4 - div(x3, (3.14 * (x2 * x2 - x1 * x1))),
        1.0 - div(2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1), np.power((x2 * x2 - x1 * x1), 2)),
        div(2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1), x2 * x2 - x1 * x1) - 900.0
    ])

    g[g >= 0] = 0
    g[g < 0] = -g[g < 0]

    f3 = np.sum(g, axis=1)[0]

    return np.array([f1, f2, f3])


population_size = 1000
num_variables = 4
num_objectives = 3
num_generations = 300
eta_crossover = 20
eta_mutation = 20
crossover_probability = 0.6
# lower_bound = [1, np.sqrt(2), np.sqrt(2), 1] # RE1
# upper_bound = [3, 3, 3, 3] # RE!
lower_bound = [55, 75, 1000, 11]  # RE5
upper_bound = [80, 110, 3000, 20]  # RE5
num_divisions = 8
polar_offset_limit = np.pi
num_max_sectors = 20
front_frequency_threshold = 0.1
niche_ratio = 0.15
monte_carlo_frequency = 4
verbose = True
log = ["hv"]
problem = re5
problem_name = "re5"
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
    # ax = fig.add_subplot(111, projection="3d")
    # ax = fig.add_subplot(111)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        calc_pareto_front(problem_name)[:, 0],
        calc_pareto_front(problem_name)[:, 1],
        calc_pareto_front(problem_name)[:, 2],
        color="red",
        alpha=0.3,
        label="Optimal Pareto Front",
    )

    ax.scatter(
        individuals[:, 0],
        individuals[:, 1],
        individuals[:, 2],
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
    plt.show()

    # hv_ref_point = np.array([-0.5, -0.5, -0.5])
    hv_ref_point = np.max(calc_pareto_front(problem_name), axis=0) + 0.5
    # hv_ref_point = np.array([1000, 1000])

    hypervolumes = model.metric("hypervolume", all_gens=True, ref=hv_ref_point)
    fig = plt.figure(figsize=(7, 7))
    plt.plot(hypervolumes)
    plt.xlabel("Iterations (t)")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume over time")
    # plt.savefig(f"images/{problem_name}_mc_nsga3_hypervolume.png", dpi=300)
    # plt.savefig(f"images/{problem_name}_nsga3_hypervolume.png", dpi=300)
    plt.show()

    path = f"result/re/{problem_name}/{model_name}"

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
    problem = RE5()
    pop_size = 500
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=20)
    n_neighbors = 20
    decomposition = None
    prob_neighbor_mating = 0.9
    sampling = FloatRandomSampling()
    cross_prob = 0.9
    crossover = SBX(prob=cross_prob, eta=eta_crossover)
    mutation = PM(prob_var=None, eta=eta_mutation)
    n_gen = 200
    hv_ref_point = np.max(calc_pareto_front(problem_name), axis=0) + 0.5

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

    res = minimize(problem, model, termination=("n_gen", n_gen), seed=1, verbose=True, save_history=True)

    # Scatter().add(res.history[0].pop.get("F")).show()
    plt.show()

    hypervolumes = MOEAD.history_hypervolume(res.history, ref=hv_ref_point)
    # print(hypervolumes)
    fig = plt.figure(figsize=(7, 7))
    plt.plot(hypervolumes)
    plt.xlabel("Iterations (t)")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume over time")
    # plt.savefig(f"images/{problem_name}_mc_nsga3_hypervolume.png", dpi=300)
    # plt.savefig(f"images/{problem_name}_nsga3_hypervolume.png", dpi=300)
    plt.show()

    path = f"result/re/{problem_name}/{model_name}"

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
    for i in range(1, 5 + 1):
        global expr
        expr = i
        run_nsga("nsga3")
        run_nsga("mc_nsga3")
    # run_moead()


def main():
    print(calc_pareto_front(problem_name))

    # plot using seaborn
    sns.set_theme(style="darkgrid")
    fig = plt.figure(figsize=(7, 7))
    # ax = fig.add_subplot(111, projection="3d")
    # ax = fig.add_subplot(111)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        calc_pareto_front(problem_name)[:, 0],
        calc_pareto_front(problem_name)[:, 1],
        calc_pareto_front(problem_name)[:, 2],
        color="red",
        alpha=0.3,
        label="Optimal Pareto Front",
    )
    plt.show()


if __name__ == "__main__":
    # run()
    main()
