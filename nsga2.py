from utils import *
import random
from typing import Callable
from population import Population
from member import Member


class NSGA2:
    def __init__(self, population_size: int, num_variables: int, num_objectives: int, objectives: list[Callable],
                 lower_bounds: np.array = None, upper_bounds: np.array = None, num_generations: int = 250,
                 tournament_size: int = 2, eta_crossover: float = 1.0, eta_mutation: float = 1.0,
                 crossover_probability: float = 0.9):
        self.population_size = population_size
        self.num_variables = num_variables
        self.num_objectives = num_objectives
        self.objectives = objectives
        self.lower_bounds = lower_bounds if lower_bounds is not None else - np.ones(num_variables)
        self.upper_bounds = upper_bounds if upper_bounds is not None else np.ones(num_variables)
        self.population: Population = self.init_population()
        self.offsprings: Population = Population([])
        self.num_generations = num_generations
        self.tournament_size = tournament_size
        self.eta_crossover = eta_crossover
        self.eta_mutation = eta_mutation
        self.crossover_probability = crossover_probability
        self.mutation_probability = 1.0 / self.num_objectives

    def create_member(self) -> Member:
        """
        Create a member of the population
        :return: The created member
        """

        # create a random chromosome in the range self.lower_bounds to self.upper_bounds which are vectors
        chromosome = self.lower_bounds + (self.upper_bounds - self.lower_bounds) * np.random.rand(self.num_variables)
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

    @classmethod
    def fast_non_dominated_sort(cls, population: Population) -> list[list]:
        """
        Fast Non-Dominated Sorting
        :return: A list of fronts
        """

        # population.reset()
        # print("population size: ", population.size)
        # counter = 0

        dominated_solutions = {x.name: set() for x in population}
        domination_count = {x.name: 0 for x in population}
        front = [set()]
        for p in population:
            for q in population:
                if p.name == q.name:
                    continue
                if p.dominates(q):
                    dominated_solutions[p.name].add(q)
                elif q.dominates(p):
                    domination_count[p.name] += 1
            if domination_count[p.name] == 0:
                p.rank = 1
                front[0].add(p)
                # counter += 1
        # print(domination_count)
        i = 1
        while front[-1]:
            next_front = set()
            for p in front[-1]:
                for q in dominated_solutions[p.name]:
                    domination_count[q.name] -= 1
                    if domination_count[q.name] == 0 and q.rank == 0:
                        q.rank = i + 1
                        next_front.add(q)
                        # counter += 1
            i += 1
            front.append(next_front)

        front = [list(x) for x in front]
        # print("counter: ", counter)
        return front

    @classmethod
    def compute_crowding_distance(cls, front: list[Member], num_objectives: int) -> None:
        """
        Compute the crowding distance for a front
        :param front: The front to compute the crowding distance for
        :param num_objectives: the number of objectives for member of fronts
        :return: The front with the crowding distance computed
        """

        n = len(front)
        if n == 0:
            return

        for member in front:
            member.crowding_distance = 0.0

        for m in range(num_objectives):
            front = sorted(front, key=lambda x: x.objective_values[m])
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            scale = front[-1].objective_values[m] - front[0].objective_values[m]
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
        mu = np.random.rand(self.num_variables)
        delta = np.zeros(self.num_variables)
        delta[mu <= 0.5] = np.power(2 * mu[mu <= 0.5], 1.0 / (self.eta_mutation + 1)) - 1
        delta[mu > 0.5] = 1 - np.power(2 * (1 - mu[mu > 0.5]), 1.0 / (self.eta_mutation + 1))
        # print(delta, mu, member.chromosome, self.lower_bounds, self.upper_bounds)
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

        child1: Member = self.create_member()
        child2 = self.create_member()
        mu = np.random.rand(self.num_variables)
        beta = np.zeros(self.num_variables)
        beta[mu <= 0.5] = np.power(2 * mu[mu <= 0.5], 1.0 / (self.eta_crossover + 1))
        beta[mu > 0.5] = np.power(1.0 / (2 * (1 - mu[mu > 0.5])), -1.0 / (self.eta_crossover + 1))
        x_1 = (parent1.chromosome + parent2.chromosome) / 2
        x_2 = np.abs((parent1.chromosome - parent2.chromosome) / 2)
        child1.chromosome = x_1 + beta * x_2
        child2.chromosome = x_1 - beta * x_2
        return self.mutate(child1), self.mutate(child2)

    def tournament(self):
        participants = random.sample(self.population.population, self.tournament_size)
        best = participants[0]
        for p in participants:
            if p > best:
                best = p
        return best

    def make_new_population(self) -> Population:
        offsprings = Population([])
        while offsprings.size < self.population_size:
            parent1, parent2 = self.tournament(), self.tournament()
            while parent1.name == parent2.name:
                parent2 = self.tournament()
            child1, child2 = self.crossover(parent1, parent2)
            offsprings += [child1, child2]
        return offsprings

    def run_generation(self) -> None:
        """
        Run the algorithm for one generation
        """

        R = self.population + self.offsprings
        # print(R.size)
        front = self.fast_non_dominated_sort(R)
        next_population = Population([])
        i = 0
        # print(fronts)
        # print("front size: ", sum(len(x) for x in front))
        while i < len(front) and next_population.size + len(front[i]) <= self.population_size:
            self.compute_crowding_distance(front[i], self.num_objectives)
            # print(front)
            next_population += front[i]
            # print(next_population.size)
            i += 1
        if i < len(front) and next_population.size < self.population_size:
            front[i] = sorted(front[i], reverse=True)
            next_population += front[i][:self.population_size - next_population.size]
        self.population = next_population
        self.offsprings = self.make_new_population()

    def run(self) -> None:
        """
        Run the algorithm for a given number of generations
        :param num_generations: The number of generations to run the algorithm for
        """

        self.fast_non_dominated_sort(self.population)
        self.offsprings = self.make_new_population()
        for i in range(self.num_generations):
            self.run_generation()
            # print(self.population)
            print(f'Generation {i + 1} done')

    def evaluate_distance_metric(self, pareto_front: np.array):
        """
        Evaluate the distance of the solutions in the population to the actual Pareto front
        """


    def evaluate_diversity_metric(self):
        """
        Evaluate the diversity of the solutions in the population
        """
        ...
