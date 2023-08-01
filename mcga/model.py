from nsga2.model import NSGA2
from nsga2.population import Population
from nsga2.moop import MOOP
from nsga2.utils import *
from itertools import product
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d


class MCGA(NSGA2):
    """
    Monte Carlo Genetic Algorithm
    This class inherits from the NSGA2 class. It will contain the following additional attributes:
        - polar_offset_limit: The limit of the polar offset
        - num_max_sectors: The maximum number of sectors to divide the polar space into
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
            polar_offset_limit: np.float64 = 2 * np.pi,
            num_max_sectors: int = 10,
            front_frequency_threshold: float = 0.1,
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
        self.polar_offset_limit = polar_offset_limit
        self.num_max_sectors = num_max_sectors
        self.cached_population: Population = Population()
        self.front_frequency_difference: float = np.inf
        self.front_frequency_threshold: float = front_frequency_threshold

        self.fig = plt.figure(figsize=(10, 10))

    def divide_planes(self, population: Population):
        """
        Create sectors in polar space,
        each sector is a tuple of (start, end) angles in n-dimensional polar space
        where n is the number of objectives
        :return: The sectors
        """

        population.to_polar()

        # min_polar_objectives = np.min(
        #     [member.polar_objective_values for member in population.population], axis=0
        # )[1:]
        #
        # max_polar_objectives = np.max(
        #     [member.polar_objective_values for member in population.population], axis=0
        # )[1:]

        # create a list of sectors
        sectors_points = []

        # divide the 2 * pi radians into num_sectors sectors randomly
        for i in range(self.moop.num_objectives - 1):
            num_sectors = random.randint(
                3 * self.num_max_sectors // 4, self.num_max_sectors
            )
            # sector = [random.uniform(max(0, min_polar_objectives[i] - EPSILON),
            #                          min(2 * np.pi, max_polar_objectives[i] + EPSILON)) for _ in range(num_sectors)]
            sector = [random.uniform(0, 2 * np.pi) for _ in range(num_sectors)]
            sector = sorted(sector)
            sector = np.array(sector)
            sectors_points.append(sector)

        sectors = []
        # now create the (start, end) tuples for each sector
        for i in range(self.moop.num_objectives - 1):
            sectors.append(
                # [(max(0, min_polar_objectives[i] - EPSILON), sectors_points[i][0])]
                [(0, sectors_points[i][0])]
                + [
                    (sectors_points[i][j], sectors_points[i][j + 1])
                    for j in range(len(sectors_points[i]) - 1)
                ]
                # + [(sectors_points[i][-1], min(2 * np.pi, max_polar_objectives[i] + EPSILON))]
                + [(sectors_points[i][-1], 2 * np.pi)]
            )

        # now rotate the sectors to create the offset
        offset = random.uniform(0, self.polar_offset_limit)
        for i in range(self.moop.num_objectives - 1):
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

    def slice_polar_space(self, population: Population = None) -> list[Population]:
        """
        Slice the polar space into sectors and assign each member of the population to a sector
        :param population: The population to slice
        :return: The sliced population in polar space
        """

        if population is None:
            population = self.population

        plane_sectors = self.divide_planes(population)

        # create the sector in polar space
        sectors = self.create_sectors(plane_sectors)

        sliced_population = [Population() for _ in range(len(sectors))]
        # divide the population into sectors
        for member in population.population:
            for i in range(len(sectors)):
                if member.is_in_sectors(sectors[i]):
                    sliced_population[i].append(member)
                    break
            else:
                print(f"Error: Member {member.polar_objective_values[1:]} not in any sector")

        sectors = [sectors[i] for i in range(len(sectors)) if len(sliced_population[i]) > 0]
        sliced_population = [slc for slc in sliced_population if len(slc) > 0]

        # self.plot_monte_carlo(sliced_population, sectors)

        return sliced_population

    def mc_nds(self, sliced_population: list[Population]) -> None:
        """
        Monte Carlo Non-Dominated Sorting
        Apply non-dominated sorting on the sliced population and assign ranks to the members
        :param sliced_population: The sliced population
        """

        for population_slice in sliced_population:
            # print("Slice:", population_slice)
            front = self.fast_non_dominated_sort(population_slice)

    @classmethod
    def crowding_distance(cls, population: Population, num_objectives: int) -> None:
        """
        Compute the crowding distance of the members in the population
        :param population: The population
        :param num_objectives: The number of objectives
        """

        # sort the population based on the front values

        max_front_value = np.max([member.front_value for member in population.population])
        min_front_value = np.min([member.front_value for member in population.population])

        for member in population.population:
            member.crowding_distance = 0.0

        # divide the population into parts based on the front values
        front_value_range = max_front_value - min_front_value
        num_parts = 10
        parts_intervals = [
            min_front_value + i * front_value_range / num_parts
            for i in range(num_parts + 1)
        ]
        parts = [[] for _ in range(num_parts)]
        for member in population.population:
            for i in range(num_parts):
                if parts_intervals[i] <= member.front_value < parts_intervals[i + 1]:
                    parts[i].append(member)
                    break

        # compute the crowding distance for each part
        for part in parts:
            cls.compute_crowding_distance(part, num_objectives)

    def run_monte_carlo_step(self, population: Population = None) -> None:
        """
        Run a Monte Carlo step
        :param population: The population to run the step on
        :return: The new population
        """

        if population is None:
            population = self.population

        # slice the population into sectors
        sliced_population = self.slice_polar_space(population)

        sliced_population = [slc for slc in sliced_population if len(slc) > 0]

        # run non-dominated sorting on the sliced population
        self.mc_nds(sliced_population)

    def compute_front_frequency_difference(
            self, population: Population = None, cached_population: Population = None
    ) -> None:
        """
        Compute the difference between the front frequencies of the cached population and the current population
        :param population: The current population
        :param cached_population: The cached population
        """

        if population is None:
            population = self.population

        if cached_population is None:
            cached_population = self.cached_population

        A = np.array([member.front_frequency for member in population])
        B = np.array([member.front_frequency for member in cached_population])

        # normalize the front frequencies
        A = A / np.linalg.norm(A, ord="fro")
        B = B / np.linalg.norm(B, ord="fro")

        self.front_frequency_difference = np.linalg.norm(A - B, ord="fro")

    def make_new_population(self) -> Population:
        """
        Make a new population from the current population
        :return: The offsprings
        """

        # assign probabilities to each member of the population based on their place in the sorted population
        probabilities = np.array([i + 1 for i in range(len(self.population))][::-1])
        probabilities = probabilities / np.sum(probabilities)

        offsprings = Population()
        while offsprings.size < self.population_size:
            # select two members from the population based on the probabilities
            selected_members = np.random.choice(
                self.population.population, size=2, p=probabilities
            )
            parent1, parent2 = selected_members[0], selected_members[1]
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            offsprings += [child1, child2]

        return offsprings

    def normalize_front_frequency(self, population: Population = None) -> None:
        """
        Normalize the front frequency of the population
        :param population: The population
        """

        if population is None:
            population = self.population

        front_frequency = np.array([member.front_frequency for member in population])
        front_frequency = front_frequency / np.linalg.norm(front_frequency, ord="fro")
        for i in range(len(population)):
            population.population[i].front_frequency = front_frequency[i]

    def run_generation(self) -> None:
        """
        Run a generation of the algorithm
        :return: None
        """

        self.offsprings = self.make_new_population()
        self.evaluate_population(self.offsprings)

        R: Population = self.population + self.offsprings
        R.reset()
        self.run_monte_carlo_step(R)
        while self.front_frequency_difference > self.front_frequency_threshold:
            cached_population = R.copy()
            self.run_monte_carlo_step(R)
            self.compute_front_frequency_difference(R, cached_population)

        self.front_frequency_difference = np.inf
        self.normalize_front_frequency(R)
        self.crowding_distance(R, self.moop.num_objectives)

        sorted_R = sorted(R.population, reverse=True)

        self.population = Population(sorted_R[: self.population_size])

    def run(self) -> None:
        """
        Run the algorithm for the given number of generations
        """
        self.plot_population_frame(0, f"gif_images/generation_{0}.png")
        self.run_monte_carlo_step()
        self.run_monte_carlo_step()
        while self.front_frequency_difference > self.front_frequency_threshold:
            self.cached_population = self.population.copy()
            self.run_monte_carlo_step()
            self.compute_front_frequency_difference()

        self.front_frequency_difference = np.inf
        self.normalize_front_frequency()
        self.crowding_distance(self.population, self.moop.num_objectives)
        self.population = Population(sorted(self.population.population, reverse=True))

        for i in range(self.num_generation):
            self.run_generation()
            print(f"Generation {i + 1} done")
            self.plot_population_frame(i + 1, f"gif_images/generation_{i + 1}.png")

    def plot_monte_carlo(self, sliced_population: list[Population], sectors) -> None:

        # plot the polar sector lines in the polar space

        dim = self.moop.num_objectives
        self.fig.clear()

        if dim == 2:
            ax = self.fig.add_subplot(projection="polar")
            for sector in sectors:
                for i in range(len(sector)):
                    ax.plot(
                        sector[0][0] * np.ones(300),
                        np.arange(0, 3, 0.01),
                        color="red",
                        linewidth=0.5,
                    )
                    ax.plot(
                        sector[0][1] * np.ones(300),
                        np.arange(0, 3, 0.01),
                        color="red",
                        linewidth=0.5,
                    )

        else:
            ax = self.fig.add_subplot(111, projection="3d")
            for sector in sectors:
                theta, phi = sector[1][0], sector[1][1]
                r = np.arange(0, 100, 0.01)
                X = r * np.sin(phi) * np.cos(theta)
                Y = r * np.sin(phi) * np.sin(theta)
                Z = r * np.cos(phi)
                ax.plot(X, Y, Z, color="red", linewidth=0.5)

        # print(len(sliced_population))

        # for slice, sector in zip(sliced_population, sectors):
        # print(sector)
        # print(*[member.polar_objective_values[1:] for member in slice.population])

        # print("\n###############################\n")

        for population in sliced_population:
            self.plot_population(ax, population)

        plt.pause(0.01)
