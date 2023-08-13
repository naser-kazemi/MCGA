import numpy as np

from .reference_point import ReferencePoint
from deap import base, tools, algorithms, creator
from deap.tools._hypervolume import hv
from emoa.utils import *
import array
import copy


class NSGA3:
    def __init__(
        self,
        problem,
        num_variables,
        num_objectives,
        num_generations,
        population_size,
        lower_bound,
        upper_bound,
        num_divisions,
        crossover_probability=0.9,
        eta_crossover=20.0,
        eta_mutation=20.0,
    ):
        self.num_objectives = num_objectives
        self.num_generations = num_generations
        self.num_divisions = num_divisions
        self.population_size = population_size
        self.toolbox = self.init_toolbox(
            problem,
            num_variables,
            lower_bound,
            upper_bound,
            crossover_probability,
            eta_crossover,
            eta_mutation,
        )
        self.population = self.toolbox.population(n=population_size)
        self.current_generation = 1

        self.stats = tools.Statistics()
        self.stats.register("pop", copy.deepcopy)
        self.result_pop = None
        self.logbook = None

    def init_toolbox(
        self,
        problem,
        num_variables,
        lower_bound,
        upper_bound,
        crossover_probability,
        eta_crossover,
        eta_mutation,
    ) -> base.Toolbox:

        creator.create("FitnessMin3", base.Fitness, weights=(-1.0,) * 3)
        creator.create(
            "Individual3", array.array, typecode="d", fitness=creator.FitnessMin3
        )

        toolbox = base.Toolbox()

        toolbox.register("evaluate", problem)
        # toolbox.register("select", self.select)
        toolbox.register("select", self.select)

        toolbox.register("attr_float", uniform, lower_bound, upper_bound, num_variables)
        toolbox.register(
            "individual", tools.initIterate, creator.Individual3, toolbox.attr_float
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register(
            "mate",
            tools.cxSimulatedBinaryBounded,
            low=lower_bound,
            up=upper_bound,
            eta=eta_crossover,
        )
        toolbox.register(
            "mutate",
            tools.mutPolynomialBounded,
            low=lower_bound,
            up=upper_bound,
            eta=eta_mutation,
            indpb=1.0 / num_variables,
        )

        toolbox.pop_size = self.population_size
        toolbox.max_gen = self.num_generations
        toolbox.mut_prob = 1 / num_variables
        toolbox.cross_prob = crossover_probability

        return toolbox

    def generate_reference_points(self):
        """
        Generate the reference points
        """

        def gen_ref_points_util(work_point, left, total, depth):
            if depth == self.num_objectives - 1:
                work_point[depth] = left / total
                ref_point = ReferencePoint(work_point.copy())
                return [ref_point]

            ref_points = []
            for i in range(left + 1):
                work_point[depth] = i / total
                ref_points += gen_ref_points_util(
                    work_point, left - i, total, depth + 1
                )
            return ref_points

        return gen_ref_points_util(
            [0] * self.num_objectives,
            self.num_divisions * self.num_objectives,
            self.num_divisions * self.num_objectives,
            0,
        )

    def find_ideal_points(self, individuals):
        """
        Find the ideal point
        :param individuals: List of individuals
        :return: Ideal point
        """

        current_ideal = [np.Inf] * self.num_objectives
        for ind in individuals:
            current_ideal = np.minimum(
                current_ideal, np.multiply(ind.fitness.wvalues, -1)
            )

        return current_ideal

    def find_extreme_points(self, individuals):
        """
        Find the extreme points
        :param individuals: List of individuals
        :return: List of extreme points
        """

        return [
            sorted(individuals, key=lambda ind: ind.fitness.wvalues[o] * -1)[-1]
            for o in range(self.num_objectives)
        ]

    def find_intercepts(self, individuals, extreme_points):
        """
        Find the intercepts of the hyperplane formed by the extreme points and the axes
        :param individuals: List of individuals
        :param extreme_points: List of extreme points
        :return: List of intercepts
        """

        if has_duplicate_individuals(individuals):
            intercepts = [
                extreme_points[m].fitness.values[m] for m in range(self.num_objectives)
            ]
        else:
            b = np.ones(self.num_objectives)
            A = [point.fitness.fitness for point in extreme_points]
            x = np.linalg.solve(A, b)
            intercepts = 1 / x

        return intercepts

    @staticmethod
    def normalize_objective(individual, m, intercepts, ideal_point, epsilon=1e-20):
        """
        Normalize the objective value of an individual
        :param individual: Individual
        :param m: Index of the objective
        :param intercepts: List of intercepts
        :param ideal_point: Ideal point
        :param epsilon: Difference threshold
        """
        if np.abs(intercepts[m] - ideal_point[m]) < epsilon:
            return individual.fitness.values[m] / epsilon
        else:
            return individual.fitness.values[m] / (intercepts[m] - ideal_point[m])

    def normalize_objectives(self, individuals, intercepts, ideal_point):
        """
        Normalize the objectives of each individual
        :param individuals: List of individuals
        :param intercepts: List of intercepts
        :param ideal_point: Ideal point
        """
        for ind in individuals:
            ind.fitness.normalized_values = [
                NSGA3.normalize_objective(ind, m, intercepts, ideal_point)
                for m in range(self.num_objectives)
            ]

    @staticmethod
    def calculate_distance(direction, point):
        k = np.dot(point, direction) / np.dot(direction, direction)
        d = np.linalg.norm(
            np.subtract(np.multiply(direction, [k] * len(direction)), point)
        )
        return d

    @staticmethod
    def associate(individuals, reference_points):
        """
        Associate each individual to a reference point
        :param individuals: List of individuals
        :param reference_points: List of reference points
        """

        for ind in individuals:
            rp_distances = [
                (rp, NSGA3.calculate_distance(ind.fitness.normalized_values, rp))
                for rp in reference_points
            ]
            min_distance_rp, min_distance = min(rp_distances, key=lambda x: x[1])
            ind.reference_point = min_distance_rp
            ind.ref_point_distance = min_distance
            min_distance_rp.associate_individual(ind)

    def niche_select(self, individuals, k):
        """
        Select k individuals from the individuals list based on the niche count
        :param individuals: List of individuals
        :param k: Number of individuals to select
        """

        if len(individuals) <= k:
            return individuals

        ideal_points = self.find_ideal_points(individuals)
        extreme_points = self.find_extreme_points(individuals)
        intercepts = self.find_intercepts(individuals, extreme_points)
        self.normalize_objectives(individuals, ideal_points, intercepts)

        reference_points = self.generate_reference_points()

        self.associate(individuals, reference_points)

        res = []
        while len(res) < k:
            min_niche_count_rp = min(reference_points, key=lambda x: x.niche_count)
            min_niche_count_rps = [
                rp
                for rp in reference_points
                if rp.niche_count == min_niche_count_rp.niche_count
            ]
            chosen_rp = random.choice(min_niche_count_rps)

            associated_individuals = chosen_rp.associated_individuals

            if associated_individuals:
                if chosen_rp.niche_count == 0:
                    sel = min(
                        chosen_rp.associated_individuals,
                        key=lambda x: x.ref_point_distance,
                    )
                else:
                    sel = random.choice(chosen_rp.associated_individuals)

                res.append(sel)
                chosen_rp.remove_associated_individual(sel)
                individuals.remove(sel)
            else:
                reference_points.remove(chosen_rp)
        return res

    def select(self, individuals, k):
        assert (
            len(individuals) >= k
        ), "Number of individuals must be greater than or equal to k"

        if k == len(individuals):
            return individuals

        fronts = tools.sortLogNondominated(individuals, len(individuals))

        limit = 0
        last_front = -1
        selection = []
        for f, front in enumerate(fronts):
            if limit + len(front) <= k:
                selection.extend(front)
                limit += len(front)
                last_front = f
            else:
                break

        selection += self.niche_select(fronts[last_front + 1], k - limit)

        print(f"Generation {self.current_generation} done.")
        self.current_generation += 1

        return selection

    def run(self, verbose=False):
        toolbox = self.toolbox
        population = toolbox.population(n=toolbox.pop_size)
        stats = self.stats
        self.result_pop, self.logbook = algorithms.eaMuPlusLambda(
            population,
            toolbox,
            mu=toolbox.pop_size,
            lambda_=toolbox.pop_size,
            cxpb=toolbox.cross_prob,
            mutpb=toolbox.mut_prob,
            ngen=toolbox.max_gen,
            stats=stats,
            verbose=verbose,
        )

    def metric(self, metric="hypervolume", log=False):
        if metric == "hypervolume":
            return self.hypervolume(self.result_pop, [11.0, 11.0], log=log)
        else:
            raise ValueError("Metric not supported")

    def hypervolume(self, population=None, ref=None, log=False):
        def hypervolume_util(population, ref=None):
            front = tools.sortLogNondominated(
                population, len(population), first_front_only=True
            )
            wobjs = np.array([ind.fitness.wvalues for ind in front]) * -1
            if ref is None:
                ref = np.max(wobjs, axis=0) + 1
            return hv.hypervolume(wobjs, ref)

        if log:
            pops = self.logbook.select("pop")
            pops_obj = [
                np.array([ind.fitness.wvalues for ind in pop]) * -1 for pop in pops
            ]
            ref = np.max([np.max(wobjs, axis=0) for wobjs in pops_obj], axis=0) + 1
            return [hypervolume_util(pop, ref) for pop in pops]
        else:
            return hypervolume_util(population, ref)
