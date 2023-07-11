from utils import *


class Member(object):
    """
    Member class to represent a member of the population.
    It will contain the following attributes:
        - chromosome: The chromosome of the member
        - objectives: The objective functions of the member
    """
    ID = 0

    def __init__(self, chromosome: list, objectives: list, name: str = str(ID)):
        self.name = name
        self.chromosome: list = chromosome
        self.objectives: list = objectives
        self.rank: int = 0
        self.distance: float = 0.0
        Member.ID += 1

    def dominates(self, other):
        """
        Check if this member dominates the other
        :param other: The other member
        :return: True if this member dominates the other, False otherwise
        """
        return all(self.objectives <= other.objectives) and any(self.objectives < other.objectives)

    def __lt__(self, other):
        return self.rank < other.rank or (self.rank == other.rank and self.distance > other.distance)

    def __eq__(self, other):
        return self.rank == other.rank and self.distance == other.distance
