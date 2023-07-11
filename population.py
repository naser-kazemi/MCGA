from utils import np
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

    def __getitem__(self, item):
        return self.population[item]

    def __setitem__(self, key, value):
        self.population[key] = value
