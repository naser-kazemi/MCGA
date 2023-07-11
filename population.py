from utils import np


class Population(object):
    def __init__(self, population: np.array):
        self.population: np.array = population
        self.shape = np.shape(population)

    @property
    def size(self) -> int:
        return self.shape[0]

    @property
    def num_objectives(self) -> int:
        return self.shape[1]
