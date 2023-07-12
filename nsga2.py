from utils import *
from population import Population
from member import Member


class NSGA2:
    def __init__(self, population_size: int, num_variables: int, num_objectives: int, num_generations: int,
                 num_sort: int, crossover_rate: float):
        self.population = self.init_population(population_size, num_objectives, num_variables)
        self.num_variables = num_variables
        self.num_objectives = num_objectives
        self.num_generations = num_generations
        self.num_sort = num_sort
        self.crossover_rate = crossover_rate

    def init_population(self, population_size: int, num_objectives: int, num_variables: int) -> Population:
        """
        Initialize the population
        :param population_size: The size of the population
        :param num_objectives: The number of objectives
        :param num_variables: The number of variables
        :return: The initialized population
        """
        population = Population([])
        for i in range(population_size):
            chromosome = np.random.rand(num_variables)
            objectives = np.random.rand(num_objectives)
            population.append(Member(objectives, chromosome))
        return population

    def fast_non_dominated_sort(self) -> list[set]:
        """
        Fast Non-Dominated Sorting
        :param num_sort: Number of sorts
        :return: A list of fronts
        """
        population_size = self.population.size

        dominated_solutions = {x.name: set() for x in self.population}
        domination_count = {x.name: 0 for x in self.population}
        front = [set()]
        for i in range(population_size):
            for j in range(population_size):
                if i == j:
                    continue
                if self.population[i].dominates(self.population[j]):
                    dominated_solutions[self.population[i].name].add(self.population[j])
                elif self.population[j].dominates(self.population[i]):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                self.population[i].rank = 1
                front[0].add(self.population[i])
        i = 0
        while front[i]:
            next_front = set()
            for member in front[i]:
                for dominated_member in dominated_solutions[member.name]:
                    domination_count[dominated_member.name] -= 1
                    if domination_count[dominated_member.name] == 0:
                        dominated_member.rank = i + 1
                        next_front.add(dominated_member)
            i += 1
            front.append(next_front)

        return front[:-1]

    def compute_crowding_distance(self, front: set[Member]) -> None:
        """
        Compute the crowding distance for a front
        :param front: The front to compute the crowding distance for
        :return: The front with the crowding distance computed
        """

        n = len(front)
        if n == 0:
            return

        for member in front:
            member.crowding_distance = 0.0

        front = list(front)

        for m in range(self.num_objectives):
            front = sorted(front, key=lambda x: x.objectives[m])
            front[0].crowding_distance = float('inf')
            front[n].crowding_distance = float('inf')

            scale = front[n - 1].objectives[m] - front[0].objectives[m]
            for i in range(1, n - 1):
                front[i].crowding_distance += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / scale

    def crossover(self, parent1: Member, parent2: Member) -> (Member, Member):
        ...
