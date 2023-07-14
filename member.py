class Member(object):
    """
    Member class to represent a member of the population.
    It will contain the following attributes:
        - chromosome: The chromosome of the member
        - objectives: The objective functions of the member
    """

    def __init__(self, chromosome: list[float], objective_values: list[float]):
        self.chromosome = chromosome
        self.objective_values = objective_values
        self.dominated_by_count: int = 0
        self.rank: int = 0
        self.crowding_distance: float = 0.0

    def dominates(self, other):
        """
        Check if this member dominates the other
        :param other: The other member
        :return: True if this member dominates the other, False otherwise
        """
        comp1 = [x <= y for x, y in zip(self.objective_values, other.objective_values)]
        comp2 = [x < y for x, y in zip(self.objective_values, other.objective_values)]
        return all(comp1) and any(comp2)

    def reset(self):
        """
        Reset the member
        :return: None
        """
        self.rank = 0
        self.crowding_distance = 0.0

    def copy(self):
        """
        Copy the member
        :return: The copied member
        """
        return Member(self.chromosome.copy(), self.objective_values.copy())

    def __gt__(self, other):
        if self.rank < other.rank:
            return True
        if self.rank == other.rank:
            return self.crowding_distance > other.crowding_distance
        return False

    def __eq__(self, other):
        return self.rank == other.rank and self.crowding_distance == other.crowding_distance

    def __repr__(self):
        return f"\nObjectives:{self.objective_values},\nChromosomes: {self.chromosome}," \
               f"\nRank: {self.rank}, Crowding Distance: {self.crowding_distance}"

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(str(self.objective_values)) + hash(str(self.chromosome))
