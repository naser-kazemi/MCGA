import numpy as np
from member import Member


class Population(object):
    def __init__(self, population: list[Member]):
        self.population: list[Member] = population
        self.fronts: list = []
        self.shape = np.shape(population)

    @property
    def size(self) -> int:
        return self.shape[0]

    @property
    def num_objectives(self) -> int:
        return self.shape[1]

    def append(self, member: Member):
        self.population.append(member)

    def __getitem__(self, item):
        return self.population[item]

    def __setitem__(self, key, value):
        self.population[key] = value

    def __add__(self, other):
        if isinstance(other, Population):
            return Population(self.population + other.population)
        return Population(self.population + other)

    def __repr__(self):
        representation = "*****\n"
        for member in self.population:
            representation += f"-->\n{member}\n"
        representation += "*****"
        return representation

    def __str__(self):
        return self.__repr__()
