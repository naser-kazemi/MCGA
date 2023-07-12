import numpy as np

from utils import *
from typing import Callable
from population import Population
from member import Member


class NSGA2:
    def __init__(self, population_size: int, num_objectives: int, objectives: list[Callable], lower_bounds: np.array,
                 upper_bounds: np.array, num_generations: int, num_sort: int, eta_crossover: float = 1.0,
                 eta_mutation: float = 1.0, crossover_probability: float = 0.9):
        self.population_size = population_size
        self.num_objectives = num_objectives
        self.objectives = objectives
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.population = self.init_population()
        self.num_generations = num_generations
        self.num_sort = num_sort
        self.eta_crossover = eta_crossover
        self.eta_mutation = eta_mutation
        self.crossover_probability = crossover_probability
        self.mutation_probability = 1.0 / self.num_objectives

    def create_member(self):
        """
        Create a member of the population
        :return: The created member
        """
        chromosome = np.random.rand(self.num_objectives)
        objective_values = np.array([objective(chromosome) for objective in self.objectives])
        return Member(objective_values, chromosome)

    def compute_objectives(self, member: Member) -> None:
        """
        Compute the objectives of the member
        :param member: The member to compute the objectives for
        :return: None
        """
        member.objective_values = np.array([objective(member.chromosome) for objective in self.objectives])

    def init_population(self) -> Population:
        """
        Initialize the population
        :return: The initialized population
        """
        population = Population([])
        for i in range(self.population_size):
            population.append(self.create_member())
        return population

    def fast_non_dominated_sort(self) -> list[list]:
        """
        Fast Non-Dominated Sorting
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

        front = [list(x) for x in front[:-1]]

        return front

    def compute_crowding_distance(self, front: list[Member]) -> None:
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

        for m in range(self.num_objectives):
            front = sorted(front, key=lambda x: x.objective_values[m])
            front[0].crowding_distance = float('inf')
            front[n].crowding_distance = float('inf')

            scale = front[n - 1].objective_values[m] - front[0].objective_values[m]
            for i in range(1, n - 1):
                front[i].crowding_distance += (front[i + 1].objective_values[m] - front[i - 1].objective_values[
                    m]) / scale

    def mutate(self, member: Member) -> Member:
        """
        Perform mutation on a member with a probability of self.mutation_probability
        :param member: The member to mutate
        :return: The mutated member
        """
        if np.random.rand() > self.mutation_probability:
            return member
        mu = np.random.rand(self.num_objectives)
        delta = np.zeros(self.num_objectives)
        delta[mu <= 0.5] = np.power(2 * mu[mu <= 0.5], 1.0 / (self.eta_mutation + 1)) - 1
        delta[mu > 0.5] = 1 - np.power(2 * (1 - mu[mu > 0.5]), 1.0 / (self.eta_mutation + 1))
        member.chromosome[mu <= 0.5] += delta[mu <= 0.5] * (member.chromosome[mu <= 0.5] - self.lower_bounds[mu <= 0.5])
        member.chromosome[mu > 0.5] += delta[mu > 0.5] * (self.upper_bounds[mu > 0.5] - member.chromosome[mu > 0.5])

        member.chromosome = np.clip(member.chromosome, self.lower_bounds, self.upper_bounds)

        return member

    def crossover(self, parent1: Member, parent2: Member) -> tuple[Member, Member]:
        """
        Crossover two parents with a probability of self.crossover_probability
        :param parent1: The first parent
        :param parent2: The second parent
        :return: The two children
        """
        if np.random.rand() > self.crossover_probability:
            return parent1, parent2

        child1 = self.create_member()
        child2 = self.create_member()
        mu = np.random.rand(self.num_objectives)
        beta = np.zeros(self.num_objectives)
        beta[mu <= 0.5] = np.power(2 * mu[mu <= 0.5], 1.0 / (self.eta_crossover + 1))
        beta[mu > 0.5] = np.power(1.0 / (2 * (1 - mu[mu > 0.5])), -1.0 / (self.eta_crossover + 1))
        x_1 = (parent1.chromosome + parent2.chromosome) / 2
        x_2 = np.abs((parent1.chromosome - parent2.chromosome) / 2)
        child1.chromosome = x_1 + beta * x_2
        child2.chromosome = x_1 - beta * x_2
        return self.mutate(child1), self.mutate(child2)

    def make_new_population(self, population: Population) -> Population:
        ...

    def run(self):
        # TODO: compute Q
        Q = []
        self.population = self.population + Q
        # compute R_t based on the population and offsprings
        fronts = self.fast_non_dominated_sort()
        next_population = Population([])
        i = 0
        while next_population.size + len(fronts[i]) <= self.population_size:
            self.compute_crowding_distance(fronts[i])
            next_population += fronts[i]
            i += 1
        fronts[i] = sorted(fronts[i], reverse=True)
        next_population += fronts[i][:self.population_size - next_population.size]
        Q = self.make_new_population(next_population)
