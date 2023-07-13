from pymoo.problems import get_problem
from pymoo.util.plotting import plot

from utils import create_parser, np
from nsga2 import NSGA2


def main():
    zdts = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6"]

    problem = get_problem(zdts[2])
    plot(problem.pareto_front(), no_fill=True, show=True)

    # get objective functions and bounds
    obj1 = problem.pareto_front()[:, 0]
    obj2 = problem.pareto_front()[:, 1]
    lower_bound = problem.xl
    upper_bound = problem.xu

    # create a new NSGA2 instance
    # create two objectives
    f1 = lambda x: x[0]
    g = lambda x: 1 + 9 * sum(x[1:]) / (x.shape[0] - 1)
    f2 = lambda x: g(x) * (1 - np.sqrt(x[0] / g(x)))

    objectives = [f1, f2]

    nsga2 = NSGA2(50, problem.n_var, problem.n_obj, objectives, num_generations=20, tournament_size=2,
                  eta_crossover=1.0, eta_mutation=1.0, crossover_probability=0.9, lower_bounds=lower_bound,
                  upper_bounds=upper_bound)
