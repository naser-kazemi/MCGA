import numpy as np

from emoa.utils import *

_EPS = sys.float_info.epsilon * 100
_REL_TOL = sys.float_info.dig - 2 if sys.float_info.dig - 2 > 0 else sys.float_info.dig


class ReferencePoint:
    """
    Reference Point class to represent a reference point.
    """

    def __init__(self, values: list[float]):
        self.fitness = np.array(values)
        self.fitness_on_hyperplane = np.array(values)
        self.niche_count = 0
        self.associated_members = []
        self.distance = 0.0

    def associate(self, member):
        self.associated_members.append(member)
        self.niche_count += 1

    def map_on_hyperplane(
        self, ideal_point: np.ndarray, intercepts: np.ndarray
    ) -> None:
        """
        Maps a reference point on a hyperplane. The hyperplane defined as x_1 + .... + x_n = 1,
        where n - dimension of objective space, x_i >= 0, i in {1,...,n}.
        :param ideal_point: The ideal point
        :param intercepts: The intercepts
        """

        np.copyto(self.fitness_on_hyperplane, self.fitness)

        self.fitness_on_hyperplane = self.fitness_on_hyperplane - ideal_point

        close_indices = np.isclose(ideal_point, intercepts, rtol=_REL_TOL, atol=_EPS)
        diff = (intercepts - ideal_point)[~close_indices]

        self.fitness_on_hyperplane[:, close_indices] /= _EPS
        self.fitness_on_hyperplane[:, ~close_indices] /= diff

    def __repr__(self) -> str:
        return "Fitness: {0}\nNormalized Fitness: {1}".format(
            self.fitness, self.fitness_on_hyperplane
        )

    def __str__(self) -> str:
        return self.__repr__()
