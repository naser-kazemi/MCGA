import numpy as np

from emoa.utils import *
from emoa import MOOP, Member, Population
from nsga2 import NSGA2
from .old_reference_point import ReferencePoint

from itertools import chain

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
    11: (1, 2),
}


class NSGA3(NSGA2):
    """
    NSGA3 algorithm impzxlementation.
    """

    def __init__(
        self,
        moop: MOOP,
        num_generation: int,
        population_size: int,
        crossover_probability: float = 0.9,
        tournament_size: int = 2,
        eta_crossover: float = 1.0,
        eta_mutation: float = 1.0,
        num_divisions_per_obj: int = 12,
    ):
        super().__init__(
            moop,
            num_generation,
            population_size,
            crossover_probability,
            tournament_size,
            eta_crossover,
            eta_mutation,
        )
        self.num_divisions_per_obj = num_divisions_per_obj
        self.reference_points: list[ReferencePoint] = []
        self.niche_counts = np.zeros(len(self.reference_points) + 1)
        self.divisions = None
        self.ideal_point = None

    def generate_reference_points(self, divisions: tuple[int]) -> None:
        """
        Generate the reference points on the hyperplane.

        The hyperplane defined as x_1 + .... + x_n = 1,
        where n - dimension of objective space, x_i >= 0, i in {1,...,n}.
        :param divisions: The number of divisions per objective
        :return: The reference points
        """

        num_objs = self.moop.num_objectives

        def compute_point_on_hyperplane(
            coord_base_vector: float, vector_coeff: tuple[float]
        ) -> list[float]:
            """
            Compute the point on the hyperplane.
            :param coord_base_vector: The base vector of coordinates
            :param vector_coeff: The vector of coefficients
            :return: The point on the hyperplane
            """
            point_on_hyperplane = [
                coord_base_vector * vector_coeff[idx] for idx in range(num_objs)
            ]

            return point_on_hyperplane

        if self.divisions is not None:
            if (
                self.divisions == divisions
                and len(self.reference_points[0]) == num_objs
            ):
                return

        step = 1 / len(divisions)

        for i in range(len(divisions)):
            coord_base_vector = (i + 1) * step
            cvh_coeffs = generate_coeff_convex_hull(num_objs, divisions[i] + 1)
            self.reference_points.extend(
                [
                    ReferencePoint(
                        compute_point_on_hyperplane(coord_base_vector, cvh_coeffs[j])
                    )
                    for j in range(len(cvh_coeffs))
                ]
            )
        self.niche_counts = np.zeros(len(self.reference_points))
        self.divisions = divisions

    @staticmethod
    def find_ideal_point(population: Population) -> np.ndarray:
        """
        Find the ideal point of the population
        :param population: The population
        """
        objective_values = np.array([member.objective_values for member in population])
        return np.min(objective_values, axis=0, initial=np.inf)

    def find_extreme_points(self, population: Population) -> np.ndarray:
        """
        Find the extreme points of the population
        :param population: The population
        :return: The extreme points
        """
        normalized_objective_values = np.array(
            [member.normalized_objective_values for member in population]
        )
        num_objs = self.moop.num_objectives

        weights = np.full(num_objs, 1e-6)
        extreme_points = np.zeros((num_objs, num_objs))

        for m in range(num_objs):
            weights[m - 1] = 1e-6
            weights[m] = 1.0

            min_normalized_objective_values = min(
                normalized_objective_values, key=lambda x: asf(x, weights)
            )
            extreme_points[m] = min_normalized_objective_values

        return extreme_points

    @staticmethod
    def find_divisions(ref_points):
        if isinstance(ref_points, int):
            divisions = (ref_points,)
        else:
            divisions = ref_points

        return divisions

    @staticmethod
    def find_intercepts(population: Population, extreme_points: np.ndarray):
        objective_values = np.array([member.objective_values for member in population])
        try:
            b = np.ones(len(extreme_points))
            A = extreme_points
            solution = np.linalg.solve(A, b, overwrite_a=True, overwrite_b=True)
            intercepts = 1 / solution
            if np.any(intercepts < 0):
                intercepts = np.max(objective_values, axis=0)

        except np.linalg.LinAlgError:
            intercepts = np.zeros(len(extreme_points))

        return intercepts

    @staticmethod
    def compute_distance(direction: np.ndarray, point: np.ndarray):
        dot_product = np.dot(direction, point)
        squared_norm = np.dot(direction, direction)

        coefficient = dot_product / squared_norm

        return np.linalg.norm((coefficient * direction) - point)

    def associate(
        self,
        population_exclude_last_front: Population,
        population_last_front: Population,
    ) -> list[dict]:
        """
        Associate each member of the population with the reference point
        :param population_exclude_last_front: The population exclude the last front
        :param population_last_front: The population of the last front
        """

        pop_size = len(population_exclude_last_front) + len(population_last_front)

        closest_ref_points = [
            {"distance": 0.0, "indices_ref_points": set()} for _ in range(pop_size)
        ]

        distances = np.zeros(len(self.reference_points))
        for idx_pop, member in enumerate(
            chain(population_exclude_last_front, population_last_front)
        ):
            for idx_ref_point, ref_point in enumerate(self.reference_points):
                distances[idx_ref_point] = self.compute_distance(
                    ref_point.fitness_on_hyperplane, member.objective_values
                )

            min_distance = np.min(distances)
            closest_ref_points[idx_pop]["distance"] = min_distance
            indices = (distances == min_distance).nonzero()[0]

            for i in indices:
                if idx_pop < len(population_exclude_last_front):
                    # self.reference_points[i].niche_count += 1
                    self.niche_counts[i] += 1
                closest_ref_points[idx_pop]["indices_ref_points"].add(i)

        return closest_ref_points

    def niche_select(
        self,
        num_select: int,
        closest_ref_points: list[dict],
        population_last_front: Population,
    ) -> list:
        """
        Select num_select members from the last front
        :param num_select: The number of members to select
        :param closest_ref_points: The closest reference points
        :param population_last_front: The population of the last front
        """
        k = 1
        indices_last_front = set(range(len(population_last_front)))
        indices_to_add = []

        type_info = np.iinfo(self.niche_counts.dtype)

        diff_len = len(closest_ref_points) - len(population_last_front)

        while k <= num_select:
            min_niche_count = self.niche_counts.min()
            indices_min = (self.niche_counts == min_niche_count).nonzero()[0]
            random_index = np.random.choice(indices_min)

            indices_pop_closest_to_ref_point = [
                diff_len + index
                for index in indices_last_front
                if random_index
                in closest_ref_points[diff_len + index]["indices_ref_points"]
            ]

            if indices_pop_closest_to_ref_point:
                if self.niche_counts[random_index] == 0:
                    index_min = min(
                        indices_pop_closest_to_ref_point,
                        key=lambda index: closest_ref_points[index]["distance"],
                    )
                    # Return to zero-based index because we choose from 'pop_last_front' in the end.
                    index_for_add = index_min - diff_len
                    indices_to_add.append(index_for_add)
                else:
                    index_for_add = (
                        random.choice(indices_pop_closest_to_ref_point) - diff_len
                    )
                    indices_to_add.append(index_for_add)

                self.niche_counts[random_index] += 1
                # Remove a member from further choosing.
                indices_last_front.remove(index_for_add)
                k += 1
            else:
                # Delete a reference point.
                self.niche_counts[random_index] = type_info.max

        return indices_to_add

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

        selection.extend(
            self.niche_select(front, self.population_size - len(selection))
        )
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


##############################
##############################
##############################
##############################
##############################


# def generate_reference_points(self):
#     """
#     Generate the reference points
#     """
#
#     def gen_ref_points_util(work_point, left, total, depth):
#         if depth == self.num_objectives - 1:
#             work_point[depth] = left / total
#             ref_point = ReferencePoint(work_point.copy())
#             return [ref_point]
#
#         ref_points = []
#         for i in range(left + 1):
#             work_point[depth] = i / total
#             ref_points += gen_ref_points_util(
#                 work_point, left - i, total, depth + 1
#             )
#         return ref_points
#
#     return gen_ref_points_util(
#         [0] * self.num_objectives,
#         self.num_divisions * self.num_objectives,
#         self.num_divisions * self.num_objectives,
#         0,
#     )
#
# def find_ideal_points(self, individuals):
#     """
#     Find the ideal point
#     :param individuals: List of individuals
#     :return: Ideal point
#     """
#
#     current_ideal = [np.Inf] * self.num_objectives
#     for ind in individuals:
#         current_ideal = np.minimum(
#             current_ideal, np.multiply(ind.fitness.wvalues, -1)
#         )
#
#     return current_ideal
#
# def find_extreme_points(self, individuals):
#     """
#     Find the extreme points
#     :param individuals: List of individuals
#     :return: List of extreme points
#     """
#
#     return [
#         sorted(individuals, key=lambda ind: ind.fitness.wvalues[o] * -1)[-1]
#         for o in range(self.num_objectives)
#     ]
#
# def find_intercepts(self, individuals, extreme_points):
#     """
#     Find the intercepts of the hyperplane formed by the extreme points and the axes
#     :param individuals: List of individuals
#     :param extreme_points: List of extreme points
#     :return: List of intercepts
#     """
#
#     if has_duplicate_individuals(individuals):
#         intercepts = [
#             extreme_points[m].fitness.values[m] for m in range(self.num_objectives)
#         ]
#     else:
#         b = np.ones(self.num_objectives)
#         A = [point.fitness.fitness for point in extreme_points]
#         x = np.linalg.solve(A, b)
#         intercepts = 1 / x
#
#     return intercepts
#
# @staticmethod
# def normalize_objective(individual, m, intercepts, ideal_point, epsilon=1e-20):
#     """
#     Normalize the objective value of an individual
#     :param individual: Individual
#     :param m: Index of the objective
#     :param intercepts: List of intercepts
#     :param ideal_point: Ideal point
#     :param epsilon: Difference threshold
#     """
#     if np.abs(intercepts[m] - ideal_point[m]) < epsilon:
#         return individual.fitness.values[m] / epsilon
#     else:
#         return individual.fitness.values[m] / (intercepts[m] - ideal_point[m])
#
# def normalize_objectives(self, individuals, intercepts, ideal_point):
#     """
#     Normalize the objectives of each individual
#     :param individuals: List of individuals
#     :param intercepts: List of intercepts
#     :param ideal_point: Ideal point
#     """
#     for ind in individuals:
#         ind.fitness.normalized_values = [
#             NSGA3.normalize_objective(ind, m, intercepts, ideal_point)
#             for m in range(self.num_objectives)
#         ]
#
# @staticmethod
# def calculate_distance(direction, point):
#     k = np.dot(point, direction) / np.dot(direction, direction)
#     d = np.linalg.norm(
#         np.subtract(np.multiply(direction, [k] * len(direction)), point)
#     )
#     return d
#
# @staticmethod
# def associate(individuals, reference_points):
#     """
#     Associate each individual to a reference point
#     :param individuals: List of individuals
#     :param reference_points: List of reference points
#     """
#
#     for ind in individuals:
#         rp_distances = [
#             (rp, NSGA3.calculate_distance(ind.fitness.normalized_values, rp))
#             for rp in reference_points
#         ]
#         min_distance_rp, min_distance = min(rp_distances, key=lambda x: x[1])
#         ind.reference_point = min_distance_rp
#         ind.ref_point_distance = min_distance
#         min_distance_rp.associate_individual(ind)
#
# def niche_select(self, individuals, k):
#     """
#     Select k individuals from the individuals list based on the niche count
#     :param individuals: List of individuals
#     :param k: Number of individuals to select
#     """
#
#     if len(individuals) <= k:
#         return individuals
#
#     ideal_points = self.find_ideal_points(individuals)
#     extreme_points = self.find_extreme_points(individuals)
#     intercepts = self.find_intercepts(individuals, extreme_points)
#     self.normalize_objectives(individuals, ideal_points, intercepts)
#
#     reference_points = self.generate_reference_points()
#
#     self.associate(individuals, reference_points)
#
#     res = []
#     while len(res) < k:
#         min_niche_count_rp = min(reference_points, key=lambda x: x.niche_count)
#         min_niche_count_rps = [
#             rp
#             for rp in reference_points
#             if rp.niche_count == min_niche_count_rp.niche_count
#         ]
#         chosen_rp = random.choice(min_niche_count_rps)
#
#         associated_individuals = chosen_rp.associated_individuals
#
#         if associated_individuals:
#             if chosen_rp.niche_count == 0:
#                 sel = min(
#                     chosen_rp.associated_individuals,
#                     key=lambda x: x.ref_point_distance,
#                 )
#             else:
#                 sel = random.choice(chosen_rp.associated_individuals)
#
#             res.append(sel)
#             chosen_rp.remove_associated_individual(sel)
#             individuals.remove(sel)
#         else:
#             reference_points.remove(chosen_rp)
#     return res
#
# def select(self, individuals, k):
#     assert (
#             len(individuals) >= k
#     ), "Number of individuals must be greater than or equal to k"
#
#     if k == len(individuals):
#         return individuals
#
#     fronts = tools.sortLogNondominated(individuals, len(individuals))
#
#     limit = 0
#     last_front = -1
#     selection = []
#     for f, front in enumerate(fronts):
#         if limit + len(front) <= k:
#             selection.extend(front)
#             limit += len(front)
#             last_front = f
#         else:
#             break
#
#     selection += self.niche_select(fronts[last_front + 1], k - limit)
#
#     print(f"Generation {self.current_generation} done.")
#     self.current_generation += 1
#
#     return selection
