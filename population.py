from member import Member


class Population(object):
    """
    Population class to represent a population of members. It is a wrapper around a list of members.
    It will contain the following attributes:
        - population: The population
    """

    def __init__(self, population: list[Member] = None):
        if population is None:
            population = []
        self.population: list[Member] = population

    def reset(self):
        for member in self.population:
            member.reset()

    @property
    def size(self) -> int:
        return len(self.population)

    def append(self, member: Member):
        self.population.append(member)

    def to_polar(self):
        for member in self.population:
            member.to_polar()

    def to_cartesian(self):
        for member in self.population:
            member.to_cartesian()

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

    def __len__(self):
        return len(self.population)
