import matplotlib.pyplot as plt
import numpy as np
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

from moead import MOEAD

# from pymoo.algorithms.moo.moead import MOEAD

from pymoo.core.problem import Problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from emoa.utils import *

A = np.array(
    [
        [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
        [-np.cos(np.pi / 4), np.sin(np.pi / 4), 0],
        [0, 0, 0],
    ]
)


class FTest2(Problem):
    def __init__(self):
        n_var = 10
        n_obj = 2
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=0,
            xl=np.array([0.0] + (n_var - 1) * [-1.0]),
            xu=np.array(n_var * [1.0]),
            type_var=np.double,
        )

    def f1(self, x):
        value = x[0]
        J1 = {j for j in range(2, self.n_var + 1) if j % 2 == 1}
        temp_value = 0.0
        for j in J1:
            temp_value += (
                x[j - 1] - np.sin(6 * np.pi * x[0] + j * np.pi / self.n_var)
            ) ** 2

        value += temp_value * 2.0 / len(J1)
        return value

    def f2(self, x):
        value = 1 - np.sqrt(x[0])
        J2 = {j for j in range(2, self.n_var + 1) if j % 2 == 0}
        temp_value = 0.0
        for j in J2:
            temp_value += (
                x[j - 1] - np.sin(6 * np.pi * x[0] + j * np.pi / self.n_var)
            ) ** 2
        value += temp_value * 2.0 / len(J2)
        return value

    def f(self, x):
        return np.array([self.f1(x), self.f2(x)])

    def pareto_set(self, *args, **kwargs):
        x1 = np.linspace(0, 1, 100)
        X = np.zeros((len(x1), self.n_var))
        X[:, 0] = x1
        for j in range(2, self.n_var + 1):
            X[:, j - 1] = np.sin(6 * np.pi * x1 + j * np.pi / self.n_var)
        return X

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.apply_along_axis(self.f, 1, x)


class FTest3(Problem):
    def __init__(self):
        n_var = 10
        n_obj = 2
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=0,
            xl=np.array([0.0] + (n_var - 1) * [-1.0]),
            xu=np.array(n_var * [1.0]),
            type_var=np.double,
        )

    def f1(self, x):
        value = x[0]
        J1 = {j for j in range(2, self.n_var + 1) if j % 2 == 1}
        temp_value = 0.0
        for j in J1:
            temp_value += (
                x[j - 1] - np.power(x[0], 0.5 * (1 + 3 * (j - 2) / (self.n_var - 2)))
            ) ** 2

        value += temp_value * 2.0 / len(J1)
        return value

    def f2(self, x):
        value = 1 - np.sqrt(x[0])
        J2 = {j for j in range(2, self.n_var + 1) if j % 2 == 0}
        temp_value = 0.0
        for j in J2:
            temp_value += (
                x[j - 1] - np.power(x[0], 0.5 * (1 + 3 * (j - 2) / (self.n_var - 2)))
            ) ** 2
        value += temp_value * 2.0 / len(J2)
        return value

    def f(self, x):
        return np.array([self.f1(x), self.f2(x)])

    def pareto_set(self, *args, **kwargs):
        J1 = {j for j in range(2, self.n_var + 1) if j % 2 == 1}
        J2 = {j for j in range(2, self.n_var + 1) if j % 2 == 0}
        x1 = np.linspace(0, 1, 100)
        X = np.zeros((len(x1), self.n_var))
        X[:, 0] = x1
        for j in range(2, self.n_var + 1):
            if j in J1:
                X[:, j - 1] = 0.8 * x1 * np.cos(6 * np.pi * x1 + j * np.pi / self.n_var)
            elif j in J2:
                X[:, j - 1] = 0.8 * x1 * np.sin(6 * np.pi * x1 + j * np.pi / self.n_var)
        return X

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.apply_along_axis(self.f, 1, x)


class FTest6(Problem):
    def __init__(self):
        n_var = 10
        n_obj = 3
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=0,
            xl=np.array([0.0, 0.0] + (n_var - 2) * [-2.0]),
            xu=np.array([1.0, 1.0] + (n_var - 2) * [2.0]),
            type_var=np.double,
        )

    def f1(self, x):
        value = np.cos(0.5 * np.pi * x[0]) * np.cos(0.5 * np.pi * x[1])
        J1 = {j for j in range(3, self.n_var + 1) if j % 3 == 1}
        temp_value = 0.0
        for j in J1:
            temp_value += (
                x[j - 1] - 2 * x[1] * np.sin(2 * np.pi * x[0] + j * np.pi / self.n_var)
            ) ** 2

        value += temp_value * 2.0 / len(J1)
        return value

    def f2(self, x):
        value = np.cos(0.5 * np.pi * x[0]) * np.sin(0.5 * np.pi * x[1])
        J2 = {j for j in range(3, self.n_var + 1) if j % 3 == 2}
        temp_value = 0.0
        for j in J2:
            temp_value += (
                x[j - 1] - 2 * x[1] * np.sin(2 * np.pi * x[0] + j * np.pi / self.n_var)
            ) ** 2
        value += temp_value * 2.0 / len(J2)
        return value

    def f3(self, x):
        value = np.sin(0.5 * np.pi * x[0])
        J3 = {j for j in range(3, self.n_var + 1) if j % 3 == 0}
        temp_value = 0.0
        for j in J3:
            temp_value += (
                x[j - 1] - 2 * x[1] * np.sin(2 * np.pi * x[0] + j * np.pi / self.n_var)
            ) ** 2
        value += temp_value * 2.0 / len(J3)
        return value

    def f(self, x):
        return np.array([self.f1(x), self.f2(x), self.f3(x)])

    def ps_func(self, x1, x2, j):
        return 2 * x2 * np.sin(2 * np.pi * x1 + j * np.pi / self.n_var)

    def pareto_set(self, *args, **kwargs):
        x1 = np.linspace(0, 1, 30)
        x2 = np.linspace(0, 1, 30)

        mesh = np.meshgrid(x1, x2)

        X = np.zeros((len(x1) * len(x2), self.n_var))
        X[:, 0] = mesh[0].flatten()
        X[:, 1] = mesh[1].flatten()
        for j in range(3, self.n_var + 1):
            X[:, j - 1] = self.ps_func(mesh[0].flatten(), mesh[1].flatten(), j)

        return X

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.apply_along_axis(self.f, 1, x)


def run():
    problem = FTest2()
    # problem = FTest3()
    # problem = FTest6()
    pop_size = 400
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=400)
    n_neighbors = 20
    decomposition = None
    prob_neighbor_mating = 0.9
    sampling = FloatRandomSampling()
    crossover = SBX(prob=0.9, eta=20)
    mutation = PM(prob_var=None, eta=20)
    n_gen = 300

    model = MOEAD(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        n_neighbors=n_neighbors,
        decomposition=decomposition,
        prob_neighbor_mating=prob_neighbor_mating,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
    )

    res = minimize(problem, model, termination=("n_gen", n_gen), seed=1, verbose=True)

    # plot x1, x2 and x3 of the final population in 3D
    sns.set_theme(style="darkgrid")
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        res.X[:, 0],
        res.X[:, 1],
        res.X[:, 2],
        c="red",
        marker="o",
        label="final population",
    )
    print(res.X.shape)
    ax.scatter(
        problem.pareto_set()[:, 0],
        problem.pareto_set()[:, 1],
        problem.pareto_set()[:, 2],
        c="blue",
        marker="o",
        s=7,
        alpha=0.3,
        label="true pareto set",
    )
    ax.set_xlabel("$x_1$", fontsize=15)
    ax.set_ylabel("$x_2$", fontsize=15)
    ax.set_zlabel("$x_3$", fontsize=15)
    plt.show()

    Scatter().add(res.F).show()
    plt.show()


if __name__ == "__main__":
    run()
