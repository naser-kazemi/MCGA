import matplotlib.pyplot as plt
import numpy as np
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

from moead import MOEAD

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


class CPFT2(Problem):
    def __init__(self):
        super().__init__(
            n_var=7,
            n_obj=3,
            n_constr=0,
            xl=np.array(7 * [0.0]),
            xu=np.array(7 * [1.0]),
            type_var=np.double,
        )

    @staticmethod
    def gamma2(xi):
        return 0.5 * (2 * xi[0] - 1) * np.power(2 * xi[1] - 1, 2)

    @staticmethod
    def cpft2(xi):
        a = np.array([xi[0], xi[0], -((2 * xi[0] - 1) ** 3)])
        b = np.array([2 * xi[1] - 1, CPFT2.gamma2(xi), 0])
        return a + A @ b

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.apply_along_axis(CPFT2.cpft2, 1, x)


class CPFT3(Problem):
    def __init__(self):
        super().__init__(
            n_var=7,
            n_obj=3,
            n_constr=0,
            xl=np.array(7 * [0.0]),
            xu=np.array(7 * [1.0]),
            type_var=np.double,
        )

    @staticmethod
    def gamma3(xi):
        return np.power(np.abs(2 * xi[1] - 1), xi[0] + 0.5)

    @staticmethod
    def cpft3(xi):
        a = np.array([xi[0], xi[0], -((2 * xi[0] - 1) ** 3)])
        b = np.array([2 * xi[1] - 1, CPFT3.gamma3(xi), 0])
        return a + A @ b

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.apply_along_axis(CPFT3.cpft3, 1, x)


class CPFT4(Problem):
    def __init__(self):
        super().__init__(
            n_var=7,
            n_obj=3,
            n_constr=0,
            xl=np.array(7 * [0.0]),
            xu=np.array(7 * [1.0]),
            type_var=np.double,
        )

    @staticmethod
    def gamma(xi):
        return np.power(np.abs(2 * xi[1] - 1), 1.0 - 0.5 * np.sin(4 * np.pi * xi[0]))

    @staticmethod
    def cpft(xi):
        a = np.array([xi[0], xi[0], -((2 * xi[0] - 1) ** 3)])
        b = np.array([2 * xi[1] - 1, CPFT4.gamma(xi), 0])
        return a + A @ b

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.apply_along_axis(CPFT4.cpft, 1, x)


class CPFT5(Problem):
    def __init__(self):
        super().__init__(
            n_var=7,
            n_obj=3,
            n_constr=0,
            xl=np.array(7 * [0.0]),
            xu=np.array(7 * [1.0]),
            type_var=np.double,
        )

    @staticmethod
    def gamma1(xi):
        a = 6.217210009329 * 1e-2
        b = 2 / 3
        s = a + 4 * b * xi[0]
        return 0.2 * np.cos(3 * np.pi * s) - s

    @staticmethod
    def gamma2(xi):
        return np.power(np.abs(2 * xi[1] - 1), 1.0 - 0.5 * np.sin(4 * np.pi * xi[0]))

    @staticmethod
    def cpft(xi):
        a = np.array([2 * xi[0], 2 * xi[0], CPFT5.gamma1(xi)])
        b = np.array([2 * xi[1] - 1, CPFT5.gamma2(xi), 0])
        return a + A @ b

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.apply_along_axis(CPFT5.cpft, 1, x)


class CPFT6(Problem):
    def __init__(self):
        super().__init__(
            n_var=7,
            n_obj=3,
            n_constr=0,
            xl=np.array(7 * [0.0]),
            xu=np.array(7 * [1.0]),
            type_var=np.double,
        )

    @staticmethod
    def gamma(xi):
        k = 2
        return np.power(np.abs(2 * (k * xi[2] - np.floor(k * xi[2])) - 1), 0.5 + xi[0])

    @staticmethod
    def cpft(xi):
        k = 2
        a = np.array([xi[0], xi[0], -((2 * xi[0] - 1) ** 3)])
        b = np.array([2 * k * xi[1], CPFT6.gamma(xi), 0])
        return a + A @ b

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.apply_along_axis(CPFT6.cpft, 1, x)


def run():
    # problem = CPFT2()
    # problem = CPFT3()
    problem = CPFT4()
    # problem = CPFT5()
    # problem = CPFT6()
    pop_size = 200
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=20)
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

    Scatter().add(res.F).show()
    plt.show()


if __name__ == "__main__":
    run()
