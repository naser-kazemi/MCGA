from typing import Callable
from utils import np, random


class MOOP:
    """
    Multi-Objective Optimization Problem (MOOP) class
    It will contain the following attributes:
        -num_variables: The number of variables
        -objectives: The objectives
        -num_objectives: The number of objectives
        -pareto_front: The pareto front
        -lower_bounds: The lower bounds of the variables
        -upper_bounds: The upper bounds of the variables
    """

    def __init__(self, num_variables: int, objectives: list[Callable], pareto_front: np.array,
                 lower_bounds: list[float],
                 upper_bounds: list[float]):
        self.num_variables = num_variables
        self.objectives = objectives
        self.num_objectives = len(self.objectives)
        self.pareto_front = pareto_front
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def evaluate(self, chromosome):
        """
        Evaluate the chromosome
        :param chromosome: The chromosome to evaluate
        :return: The objective values
        """
        return [objective(chromosome) for objective in self.objectives]

    def generate_chromosome(self):
        """
        Generate a random chromosome
        :return: The random chromosome
        """
        return [random.uniform(self.lower_bounds[i], self.upper_bounds[i]) for i in range(self.num_variables)]
