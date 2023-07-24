from nsga2 import NSGA2
from member import Member
from population import Population
from moop import MOOP
from utils import *
from itertools import product


class MCGA(NSGA2):
    """
    Monte Carlo Genetic Algorithm
    This class inherits from the NSGA2 class. It will contain the following additional attributes:
        - polar_offset_limit: The limit of the polar offset
        - num_max_sectors: The maximum number of sectors to divide the polar space into
    """

    def __init__(self, moop: MOOP, num_generation: int, population_size: int, crossover_probability: float = 0.9,
                 tournament_size: int = 2, eta_crossover: float = 1.0, eta_mutation: float = 1.0,
                 polar_offset_limit: np.float64 = 2 * np.pi, num_max_sectors: int = 10):
        super().__init__(moop, num_generation, population_size, crossover_probability, tournament_size, eta_crossover,
                         eta_mutation)
        self.polar_offset_limit = polar_offset_limit
        self.num_max_sectors = num_max_sectors

    def divide_planes(self):
        """
        Create sectors in polar space,
        each sector is a tuple of (start, end) angles in n-dimensional polar space
        where n is the number of objectives
        :return: The sectors
        """

        # create a list of sectors
        sectors_points = []

        num_sectors = random.randint(3 * self.num_max_sectors // 4, self.num_max_sectors)

        # divide the 2 * pi radians into num_sectors sectors randomly
        for i in range(self.moop.num_objectives):
            sector = [random.uniform(0, 2 * np.pi) for _ in range(num_sectors)]
            sector = sorted(sector)
            sector = np.array(sector)
            sectors_points.append(sector)

        sectors = []
        # now create the (start, end) tuples for each sector
        for i in range(self.moop.num_objectives):
            sectors.append([(0, sectors_points[i][0])] + [(sectors_points[i][j], sectors_points[i][j + 1]) for j in
                                                          range(len(sectors_points[i]) - 1)] + [
                               (sectors_points[i][-1], 2 * np.pi)])

        # now rotate the sectors to create the offset
        offset = random.uniform(0, self.polar_offset_limit)
        for i in range(self.moop.num_objectives):
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
        sectors = list(product(*plane_sectors))

        return sectors

    def slice_polar_space(self, population: Population = None):
        """
        Slice the polar space into sectors and assign each member of the population to a sector
        :param population: The population to slice
        :return: The sliced population in polar space
        """

        if population is None:
            population = self.population

        plane_sectors = self.divide_planes()

        # create the sector in polar space
        sectors = self.create_sectors(plane_sectors)

        sliced_population = [Population() for _ in range(len(sectors))]
        # divide the population into sectors
        for member in population.population:
            for i in range(len(sectors)):
                if member.is_in_sectors(sectors[i]):
                    sliced_population[i].append(member)
                    break

        return sliced_population
