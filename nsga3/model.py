from emoa.utils import *
from emoa import MOOP, Member, Population
from nsga2 import NSGA2
from .reference_point import ReferencePoint

"""
Division of axis when need to create reference points each axis divided on n parts.
Keys are number of objectives.
Values are p parameter in the original algorithm.
"""
DIVISIONS_AXIS = {
    2: (4,),
    3: (10,),
    4: (8,),
    5: (6,),
    6: (5,),
    7: (4,),
    8: (2, 3),
    9: (2, 3),
    10: (2, 3),
    11: (1, 2)
}


class NSGA3(NSGA2):
    """
    NSGA3 algorithm implementation.
    """

    def __init__(self,
                 moop: MOOP,
                 num_generation: int,
                 population_size: int,
                 crossover_probability: float = 0.9,
                 tournament_size: int = 2,
                 eta_crossover: float = 1.0,
                 eta_mutation: float = 1.0, num_divisions_per_obj: int = 12):
        super().__init__(moop, num_generation, population_size, crossover_probability, tournament_size,
                         eta_crossover, eta_mutation)
        self.num_divisions_per_obj = num_divisions_per_obj
        self.reference_points: list[ReferencePoint] = []

    def generate_reference_points(self) -> list[ReferencePoint]:
        """
        Generate the reference points.
        :return: The reference points
        """

        num_objs = self.moop.num_objectives
        num_divisions_per_obj = self.num_divisions_per_obj

        def gen_refs_recursive(work_point, num_objs, left, total, depth) -> list[ReferencePoint]:
            if depth == num_objs - 1:
                work_point[depth] = left / total
                ref = ReferencePoint(work_point.copy())
                return [ref]
            else:
                res = []
                for i in range(left):
                    work_point[depth] = i / total
                    res.extend(gen_refs_recursive(work_point, num_objs, left - i, total, depth + 1))
                return res

        return gen_refs_recursive([0] * num_objs, num_objs, num_objs * num_divisions_per_obj,
                                  num_objs * num_divisions_per_obj, 0)

    def find_ideal_point(self, population: Population) -> list[float]:
        """
        Find the ideal point of the population
        :param population: The population
        :return: The ideal point
        """

        current_ideal_point = [float("inf")] * len(population[0].objective_values)
        for member in population:
            current_ideal_point = np.minimum(current_ideal_point, member.objective_values)
        return current_ideal_point

    def find_extreme_points(self, population: Population = None) -> list[list[float]]:
        """
        Find the extreme points of the population
        :param population: The population
        :return: The extreme points
        """

        if population is None:
            population = self.population

        num_objs = len(population[0].objective_values)
        extreme_points = [[0] * num_objs for _ in range(num_objs)]
        for member in population:
            for i in range(num_objs):
                if member.objective_values[i] > extreme_points[i][i]:
                    extreme_points[i] = member.objective_values.copy()
        return extreme_points

    def construct_hyperplane(self, population: Population, extreme_points: list[list[float]]) -> list[float]:
        """
        calculate the intercepts of the hyperplane constructed by the extreme points
        :param extreme_points: The extreme points
        :param population: The population
        :return: The constructed hyperplane
        """

        num_objs = self.moop.num_objectives

        if has_duplicate_member(population):
            intercepts = [extreme_points[i][i] for i in range(num_objs)]
        else:
            b = np.ones(num_objs)
            A = np.array(extreme_points)
            x = np.linalg.solve(A, b)
            intercepts = 1 / x
            intercepts = intercepts.tolist()
        return intercepts

    def normalize_objectives(self, population: Population, intercepts: list[float], ideal_point: list[float]) -> None:
        """
        Normalize the objectives of the population with the given intercepts and ideal point
        :param intercepts: The intercepts
        :param ideal_point: The ideal point
        :param population: The population
        :return: None
        """

        num_objs = self.moop.num_objectives

        def normalize_objective(member, m, epsilon=1e-10):
            """
            Normalize the objective of a member
            """
            if abs(intercepts[m] - ideal_point[m]) > epsilon:
                member.normalized_objective_values[m] = (member.objective_values[m] - ideal_point[m]) / (
                        intercepts[m] - ideal_point[m])
            else:
                member.normalized_objective_values[m] = member.objective_values[m] / epsilon

        for member in population:
            for m in range(num_objs):
                normalize_objective(member, m)

    def associate(self, population: Population, reference_points: list[ReferencePoint]) -> None:
        """
        Associate each member with a reference point
        :param reference_points: The reference points
        :param population: The population
        """

        for member in population:
            rp_dists = [(rp, rp.perpendicular_distance(member.normalized_objective_values)) for rp in reference_points]
            best_rp, best_dist = min(rp_dists, key=lambda x: x[1])
            member.reference_point = best_rp
            member.reference_point_distance = best_dist
            best_rp.associations_count += 1
            best_rp.associations.append(member)

    def niche_select(self, population: Population, k: int) -> Population:
        """
        Select the k nearest neighbors of a member
        :param population: The population
        :param k: The number of neighbors to select
        :return: The selected neighbors
        """

        if len(population) == k:
            return population

        ideal_point = self.find_ideal_point(population)
        extreme_points = self.find_extreme_points(population)
        intercepts = self.construct_hyperplane(population, extreme_points)
        self.normalize_objectives(population, intercepts, ideal_point)

        reference_points = self.generate_reference_points()
        self.associate(population, reference_points)

        res = Population()
        while len(res) < k:
            min_assoc_rp = min(reference_points, key=lambda x: x.associations_count)
            min_assoc_rps = [rp for rp in reference_points if rp.associations_count == min_assoc_rp.associations_count]
            chosen_rp = random.choice(min_assoc_rps)

            associated_members = chosen_rp.associations

            if associated_members:
                if chosen_rp.associations_count == 0:
                    sel = min(associated_members, key=lambda x: x.reference_point_distance)
                else:
                    sel = random.choice(associated_members)

                res.append(sel)
                chosen_rp.associations.remove(sel)
                chosen_rp.associations_count += 1
            else:
                reference_points.remove(chosen_rp)

        return res

    def select(self, population: Population):
        """
        The selection operator of the algorithm
        :param population: The population
        """
        assert len(population) >= self.population_size

        if len(population) == self.population_size:
            return population

        fronts = self.fast_non_dominated_sort(population)

        selection = Population()
        front = fronts[0]
        for f in fronts:
            if len(selection) + len(f) > self.population_size:
                front = f
                break
            selection.extend(f)

        selection.extend(self.niche_select(front, self.population_size - len(selection)))
        return selection

    def run_generation(self):
        """
        Run a generation of the algorithm
        :return: None
        """
        self.offsprings = self.make_new_population()
        self.evaluate_population(self.offsprings)

        r = self.population + self.offsprings
        next_population = self.select(r)

        self.population = next_population
