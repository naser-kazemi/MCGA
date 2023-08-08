from .moop import MOOP
from .member import Member
from .population import Population
from .utils import *


class GAModule(object):
    """
    Genetic Algorithm module to perform the genetic operations.
    """

    def __init__(self, moop: MOOP,
                 num_generation: int,
                 population_size: int,
                 crossover_probability: float = 0.9):
        self.moop = moop
        self.num_generation = num_generation
        self.population_size = population_size
        self.crossover_probability = crossover_probability

    def create_member(self) -> Member:
        """
        Create a member of the population
        :return: The created member
        """
        ...

    def evaluate_population(self, population: Population = None) -> None:
        """
        Evaluate the population
        :param population: The population to evaluate
        :return: None
        """
        ...

    def init_population(self):
        """
        Initialize the population
        :return: The initialized population
        """
        ...

    def crossover(self, parent1: Member, parent2: Member) -> tuple[Member, Member]:
        """
        Crossover two parents with a probability of self.crossover_probability
        :param parent1: The first parent
        :param parent2: The second parent
        :return: The two children
        """
        ...

    def mutate(self, member: Member) -> Member:
        """
        Perform mutation on a member with a probability of self.mutation_probability
        :param member: The member to mutate
        :return: The mutated member
        """
        ...

    def make_new_population(self) -> Population:
        """
        Make a new population from the current population
        :return: The offsprings
        """
        ...

    def run_generation(self) -> None:
        """
        Run the algorithm for one generation
        """
        ...

    def run(self) -> None:
        """
        Run the algorithm for a given number of generations
        """
        ...

    def plot_population_frame(self, generation, lim_ratio, filename: str) -> None:
        """
        Plot the population and save the plot to a file as a frame for a gif
        :param generation: The generation number
        :param lim_ratio: The ratio of the limits of the plot
        :param filename: The name of the file to save the plot to
        """
        ...
