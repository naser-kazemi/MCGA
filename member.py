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

    def __init__(self, objective_values: np.array, chromosome: np.array = np.array([]), name: str = ""):
        temp_name = get_id()
        self.name = name if name != "" else f"Member {temp_name}"
        self.objective_values: np.array = objective_values
        self.chromosome: np.array = chromosome
        self.rank: int = 0
        self.crowding_distance: float = 0.0

    def dominates(self, other):
        """
        Check if this member dominates the other
        :param other: The other member
        :return: True if this member dominates the other, False otherwise
        """
        return np.all(self.objective_values <= other.objective_values) and np.any(
            self.objective_values < other.objective_values)

    def __lt__(self, other):
        return self.rank > other.rank or (self.rank == other.rank and self.crowding_distance < other.crowding_distance)

    def __eq__(self, other):
        return self.rank == other.rank and self.crowding_distance == other.crowding_distance

    def __repr__(self):
        return f"{self.name},\n{self.objective_values},\n{self.chromosome},\n{self.rank}, {self.crowding_distance}"

    def __str__(self):
        return self.__repr__()
