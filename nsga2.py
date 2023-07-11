from utils import *
from population import Population


class NSGA2:
    def __init__(self):
        ...

    def fnd_sort(self, population: Population, num_sort: int):
        """
        Fast Non-Dominated Sorting
        :param population: Population object to be sorted
        :param num_sort: Number of sorts
        :return:
        """
        population_size = population.size
        population_objectives = population.num_objectives

        # Initialize the front number to 0

