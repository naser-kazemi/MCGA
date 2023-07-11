from utils import *
from population import Population


class NSGA2:
    def __init__(self, population: Population, num_generations: int, num_sort: int, crossover_rate: float):
        self.population = population
        self.num_generations = num_generations
        self.num_sort = num_sort
        self.crossover_rate = crossover_rate

    def fast_non_dominated_sort(self, num_sort: int):
        """
        Fast Non-Dominated Sorting
        :param num_sort: Number of sorts
        :return:
        """
        population_size = self.population.size
        population_objectives = self.population.num_objectives

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
