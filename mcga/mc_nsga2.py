import array
import copy
from itertools import product, chain
from operator import attrgetter

from deap.tools.emo import assignCrowdingDist
from scipy.spatial import Delaunay

from emoa.utils import np, random, vector_to_polar, vector_to_cartesian
from nsga2 import NSGA2

from deap import base, creator, tools, algorithms


class MCNSGA2(NSGA2):
    def __init__(
            self,
            problem,
            num_variables,
            num_objectives,
            num_generations,
            population_size,
            lower_bound,
            upper_bound,
            crossover_probability=0.9,
            eta_crossover=20.0,
            eta_mutation=20.0,
            log=None,
            nd="log",
            verbose=False,
            polar_offset_limit: np.float64 = 2 * np.pi,
            num_max_sectors: int = 10,
            front_frequency_threshold: float = 0.1,
            niche_ratio: float = 0.1,
            monte_carlo_frequency: int = 5,
            polar_scale: float = 1000.0,
    ):
        super().__init__(
            problem=problem,
            num_variables=num_variables,
            num_objectives=num_objectives,
            num_generations=num_generations,
            population_size=population_size,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            crossover_probability=crossover_probability,
            eta_crossover=eta_crossover,
            eta_mutation=eta_mutation,
            log=log,
            nd=nd,
            verbose=verbose,
        )

        self.polar_offset_limit = polar_offset_limit
        self.num_max_sectors = num_max_sectors
        self.front_frequency_threshold = front_frequency_threshold
        self.niche_ratio = niche_ratio
        self.monte_carlo_frequency = monte_carlo_frequency
        self.scale = polar_scale

        self.nd_sort = self.init_ndsort(nd)

    def init_ndsort(self, nd):
        if nd == "standard":
            return tools.sortNondominated
        elif nd == "log":
            return tools.sortLogNondominated
        else:
            raise Exception(
                "The choice of non-dominated sorting "
                "method '{0}' is invalid.".format(nd)
            )

    def create_individual_class(self):
        creator.create(
            "FitnessMin",
            base.Fitness,
            weights=(-1.0,) * self.num_objectives,
            front_freq=np.zeros(self.population_size * 2, dtype=np.float64),
        )
        creator.create(
            "Individual", array.array, typecode="d", fitness=creator.FitnessMin
        )

    def evaluate(self, population):
        for ind in population:
            ind.fitness.values = self.problem(ind)

    def run(self):
        population = self.toolbox.population(n=self.population_size)
        self.evaluate(population)

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + self.stats.fields

        self.stats.compile(population)

        for g in range(1, self.num_generations + 1):
            self.evaluate(population)
            offspring = MCNSGA2.varOr(population, self.toolbox, self.population_size, self.crossover_probability,
                                      1.0 / self.num_variables)

            # combine offspring and population
            population = population + offspring
            self.evaluate(population)
            chosen = self.select(population, self.population_size)
            population = chosen

            self.stats.compile(population)

            self.print_stats(chosen)

        self.result_pop = population
        self.logbook = logbook

    def select(self, individuals, k):
        ranks, scores, sorted_ids, point_fronts = self.mc_sort(individuals)
        chosen = [individuals[i] for i in sorted_ids[:k]]
        return chosen

    def mc_sort(self, individuals):
        mc_samples = 0
        slice_radius = 100

        num_individuals = len(individuals)
        for i in range(num_individuals):
            individuals[i].fitness.idx = i

        point_fronts = np.zeros((num_individuals, num_individuals))
        norm_point_fronts = np.zeros((num_individuals, num_individuals))

        min_mc_samples = 10
        max_mc_samples = 1000
        avg_diff = np.inf

        while ((avg_diff > self.front_frequency_threshold) or (mc_samples < min_mc_samples)) and (
                mc_samples < max_mc_samples):
            cr = np.random.rand(self.num_objectives - 1)
            ora = np.random.rand(self.num_objectives - 1)

            start_angle = self.polar_offset_limit[0] + ora * (self.polar_offset_limit[1] - self.polar_offset_limit[0])
            slice_count = self.num_max_sectors[0] + np.round(cr * (self.num_max_sectors[1] - self.num_max_sectors[0]))
            rad_per_slice = (self.polar_offset_limit[1] - self.polar_offset_limit[0]) / slice_count

            slices = []
            for a, sc, r in zip(start_angle, slice_count, rad_per_slice):
                slices.append([(a + i * r, a + (i + 1) * r) for i in range(int(sc))])

            sectors = list(product(*slices))
            prev_point_fronts = point_fronts.copy()
            for s in sectors:
                points_polar = product(*s)
                poly = np.array(
                    [[0.0] * self.num_objectives] + [vector_to_cartesian(slice_radius, p) for p in points_polar])

                point = np.array([ind.fitness.values for ind in individuals])

                in_mask = Delaunay(poly).find_simplex(point)
                slice_points_ids = np.where(in_mask != -1)[0]

                if len(slice_points_ids) != 0:
                    curr_inds = [individuals[i] for i in slice_points_ids]
                    fronts = self.nd_sort(curr_inds, len(curr_inds))

                    for f in range(len(fronts)):
                        p_ids = np.array([ind.fitness.idx for ind in fronts[f]])
                        point_fronts[p_ids, f] = point_fronts[p_ids, f] + 1

            new_norm_point_fronts = (point_fronts / np.linalg.norm(point_fronts, ord="fro"))
            if mc_samples > 0:
                prev_point_fronts = (prev_point_fronts / np.linalg.norm(prev_point_fronts, ord="fro"))
            diffs = np.linalg.norm(new_norm_point_fronts - prev_point_fronts, ord="fro")
            avg_diff = diffs
            norm_point_fronts = new_norm_point_fronts
            mc_samples += 1

        print(f"Computed fronts (#{mc_samples} hue wheel samplings, {avg_diff})")
        sorted_ids = np.lexsort(-point_fronts.T[::-1])
        ranks = np.zeros(len(individuals), dtype=np.int64)
        ranks[sorted_ids] = np.arange(len(individuals))
        scores = (len(individuals) - ranks) / len(individuals)
        sorted_ids = ranks.argsort()

        scores = scores / np.sum(scores)

        return ranks, scores, sorted_ids, point_fronts
