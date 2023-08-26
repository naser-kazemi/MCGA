import array
import copy
from itertools import product, chain
from operator import attrgetter

from deap.tools.emo import assignCrowdingDist

from emoa.utils import np, random, vector_to_polar
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

    def create_individual_class(self):
        creator.create(
            "FitnessMin",
            base.Fitness,
            weights=(-1.0,) * self.num_objectives,
            polar_coords=np.zeros(self.num_objectives, dtype=np.float64),
            front_freq=np.zeros(self.population_size * 2, dtype=np.float64),
        )
        creator.create(
            "Individual", array.array, typecode="d", fitness=creator.FitnessMin
        )

    def divide_planes(self):
        """
        Create sectors in polar space,
        each sector is a tuple of (start, end) angles in n-dimensional polar space
        where n is the number of objectives
        :return: The sectors
        """

        # create a list of sectors
        sectors_points = []

        # divide the 2 * pi radians into num_sectors sectors evenly
        for i in range(self.num_objectives - 1):
            sector = np.linspace(0, 2 * np.pi, self.num_max_sectors + 1)
            sectors_points.append(sector)

        sectors = []
        # now create the (start, end) tuples for each sector
        for i in range(self.num_objectives - 1):
            sectors.append(
                [
                    (sectors_points[i][j], sectors_points[i][j + 1])
                    for j in range(len(sectors_points[i]) - 1)
                ]
            )

        # now rotate the sectors to create the offset
        offset = random.uniform(0, self.polar_offset_limit)
        for i in range(self.num_objectives - 1):
            sectors[i] = [(x + offset, y + offset) for x, y in sectors[i]]

        return sectors

    @classmethod
    def create_sectors(cls, plane_sectors):
        """
        Create sectors in polar space,
        each sector is list of a tuples of (start, end) angles in n-dimensional polar space
        where n is the number of objectives
        The sectors are cartesian product of the plane sectors
        :return: The sectors
        """

        # create the cartesian product of the plane sectors
        sectors = np.array(list(product(*plane_sectors)))

        return sectors

    @staticmethod
    def convert_to_polar(individuals):
        """
        Convert the individuals in the population to polar coordinates
        :return: None
        """
        for ind in individuals:
            ind.fitness.polar_coords = vector_to_polar(ind.fitness.values)[1]

    @staticmethod
    def is_in_sector(sector, individual):
        is_in_bounds = True
        for x, (start, end) in zip(individual.fitness.polar_coords[1:], sector):
            if end <= 2 * np.pi:
                in_this_sector = start <= x < end
            else:
                in_this_sector = start <= x < 2 * np.pi or 0 <= x < end - 2 * np.pi
            is_in_bounds = is_in_bounds and in_this_sector

        return is_in_bounds

    def slice_polar_space(self, individuals):
        """
        Slice the polar space into sectors and assign each member of the population to a sector
        :param individuals: The population to slice
        :return: The sliced population in polar space
        """

        plane_sectors = self.divide_planes()
        sectors = self.create_sectors(plane_sectors)

        sliced_population = [[] for _ in range(len(sectors))]
        # divide the population into sectors
        for ind in individuals:
            for i in range(len(sectors)):
                if self.is_in_sector(sectors[i], ind):
                    sliced_population[i].append(ind)
                    break
            else:
                print(f"Error: Member {ind.fitness.polar_coords[1:]} not in any sector")

        sliced_population = [slc for slc in sliced_population if len(slc) > 0]

        return sliced_population

    def mc_nds(self, sliced_population) -> None:
        """
        Monte Carlo Non-Dominated Sorting
        Apply non-dominated sorting on the sliced population and assign ranks to the members
        :param sliced_population: The sliced population
        """

        for population in sliced_population:
            fronts = self.nd_sort(population, len(population))
            for i, front in enumerate(fronts):
                for ind in front:
                    ind.fitness.front_freq[i] += 1

    def monte_carlo_step(self, individuals):

        sliced_population = self.slice_polar_space(individuals)
        self.mc_nds(sliced_population)

    @staticmethod
    def compute_front_frequency_diff(individuals, cached_individuals):
        """
        Compute the difference between the front frequencies of the individuals
        :param individuals: The individuals to compare
        :param cached_individuals: The cached individuals to compare
        """

        a = np.array([ind.fitness.front_freq for ind in individuals])
        b = np.array([ind.fitness.front_freq for ind in cached_individuals])

        a = a / np.linalg.norm(a, ord="fro")
        b = b / np.linalg.norm(b, ord="fro")

        return np.linalg.norm(a - b, ord="fro")

    @staticmethod
    def front_value(individual):
        # value = 0.0
        # c = 1.0
        # for ff in individual.fitness.front_freq:
        #     value += ff * c
        #     c *= 0.8
        # return value

        return " ".join([str(x) for x in individual.fitness.front_freq])

    def mc_select(self, individuals):
        self.convert_to_polar(individuals)
        cached_individuals = copy.deepcopy(individuals)
        self.monte_carlo_step(individuals)

        while (
                self.compute_front_frequency_diff(individuals, cached_individuals)
                > self.front_frequency_threshold
        ):
            cached_individuals = copy.deepcopy(individuals)
            self.monte_carlo_step(individuals)

        sorted_individuals = sorted(individuals, key=self.front_value, reverse=True)

        return sorted_individuals

    def select(self, individuals, k):
        """
        Select the individuals to survive to the next generation
        :param individuals: The individuals to select from
        :param k: The number of individuals to select
        :return: The selected individuals
        """

        if (self.current_generation % self.monte_carlo_frequency) != 1 and (
                self.current_generation < self.num_generations):
            pareto_fronts = self.nd_sort(individuals, k)

            for front in pareto_fronts:
                assignCrowdingDist(front)

            chosen = list(chain(*pareto_fronts[:-1]))
            k = k - len(chosen)
            if k > 0:
                sorted_front = sorted(
                    pareto_fronts[-1],
                    key=attrgetter("fitness.crowding_dist"),
                    reverse=True,
                )
                chosen.extend(sorted_front[:k])

            self.print_stats(chosen=chosen)

            return chosen

        for ind in individuals:
            ind.fitness.values = tuple([x * self.scale for x in ind.fitness.values])

        mc_sorted = self.mc_select(individuals)
        chosen = mc_sorted[: self.population_size]

        for ind in individuals:
            ind.fitness.values = tuple([x / self.scale for x in ind.fitness.values])

        self.print_stats(chosen=chosen)

        return chosen
