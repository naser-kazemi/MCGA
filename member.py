import numpy as np

ID = 0


def get_id():
    global ID
    ID += 1
    return ID


class Member(object):
    """
    Member class to represent a member of the population.
    It will contain the following attributes:
        - chromosome: The chromosome of the member
        - objectives: The objective functions of the member
    """

    def __init__(self, objectives: np.array, chromosome: np.array = np.array([]), name: str = ""):
        temp_name = get_id()
        self.name = name if name != "" else f"Member {temp_name}"
        self.chromosome: np.array = chromosome
        self.objectives: np.array = objectives
        self.rank: int = 0
        self.crowding_distance: float = 0.0

    def dominates(self, other):
        """
        Check if this member dominates the other
        :param other: The other member
        :return: True if this member dominates the other, False otherwise
        """
        return np.all(self.objectives <= other.objectives) and np.anyany(self.objectives < other.objectives)

    def __lt__(self, other):
        for i in range(len(self.objectives)):
            if self.objectives[i] < other.objectives[i]:
                return True
            if self.objectives[i] > other.objectives[i]:
                return False
        return False

    def __eq__(self, other):
        return np.all(self.objectives == other.objectives)

    def less_than(self, other):
        return self.rank < other.rank or (self.rank == other.rank and self.crowding_distance > other.crowding_distance)

    def equals(self, other):
        return self.rank == other.rank and self.crowding_distance == other.crowding_distance

    def less_than_equals(self, other):
        return self.less_than(other) or self.equals(other)

    def greater_than(self, other):
        return not self.less_than_equals(other)

    def greater_than_equals(self, other):
        return not self.less_than(other)

    def __repr__(self):
        return f"{self.name},\n{self.objectives},\n{self.chromosome},\n{self.rank}, {self.crowding_distance}"

    def __str__(self):
        return self.__repr__()
