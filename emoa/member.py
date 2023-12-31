from emoa.utils import np, vector_to_polar, vector_to_cartesian, random

ID = 0.0


class Member(object):
    """
    Member class to represent a member of the population.
    It will contain the following attributes:
        - chromosome: The chromosome of the member
        - objectives: The objective functions of the member
        - dominated_by_count: The number of members that dominate this member
        - rank: The rank of the member
        - crowding_distance: The crowding distance of the member
    """

    def __init__(self, chromosome: list[float], objective_values: list[float]):
        self.chromosome = chromosome
        self.objective_values = objective_values.copy()
        self.normalized_objective_values = objective_values.copy()
        self.polar_objective_values = [0 for _ in range(len(objective_values))]
        self.dominated_by_count: int = 0
        self._rank: int = 0
        self.front_frequency = []
        self.crowding_distance: float = 0.0
        self.reference_point = None
        self.reference_point_distance: int = 0

        global ID
        self.id = ID * 1000 + random.randint(0, 1000) + 2 * ID + 0.5 * ID**2
        ID += 5.0

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
        self._rank = 0
        self.front_frequency = [0 for _ in self.front_frequency]
        self.crowding_distance = 0.0

    def copy(self):
        """
        Copy the member
        :return: The copied member
        """
        member = Member(self.chromosome.copy(), self.objective_values.copy())
        member.front_frequency = self.front_frequency.copy()
        return member

    def to_polar(self):
        """
        Convert the objective values to polar coordinates
        :return: The polar coordinates
        """

        r, theta = vector_to_polar(np.array(self.objective_values))

        self.polar_objective_values = [r] + theta.tolist()

    def to_cartesian(self):
        """
        Convert the objective values to cartesian coordinates
        :return: The cartesian coordinates
        """

        r, theta = self.polar_objective_values[0], self.polar_objective_values[1:]
        theta = np.array(theta)
        self.objective_values = vector_to_cartesian(r, theta).tolist()

    def is_in_sectors(self, sector):
        """
        Check if the member is in the sectors
        :param sector: The sector
        :return: True if the member is in the sectors, False otherwise
        """

        is_in_bounds = True
        for x, (start, end) in zip(self.polar_objective_values[1:], sector):
            if end <= 2 * np.pi:
                in_this_sector = start <= x < end
            else:
                in_this_sector = start <= x < 2 * np.pi or 0 <= x < end - 2 * np.pi
            is_in_bounds = is_in_bounds and in_this_sector

        return is_in_bounds

    @property
    def rank(self):
        """
        Get the rank of the member
        :return: The rank
        """
        return self._rank

    @rank.setter
    def rank(self, value):
        """
        Update the rank and front frequency of the member
        :return: None
        """
        self._rank = value
        self.front_frequency[self._rank] += 1

    # def __gt__(self, other):
    #     if self._rank < other.rank:
    #         return True
    #     if self._rank == other.rank:
    #         return self.crowding_distance > other.crowding_distance
    #     return False

    # def __gt__(self, other):
    #     for x, y in zip(self.front_frequency, other.front_frequency):
    #         if x > y:
    #             return True
    #         if x < y:
    #             return False
    #     return False

    @property
    def front_value(self):
        front_value = 0.0
        c = 1.0
        for i in range(len(self.front_frequency)):
            front_value += self.front_frequency[i] * c
            c *= 0.8
        return front_value

    def __gt__(self, other):

        # print(self.front_value, self.front_value - other.front_value)
        if (self.front_value - other.front_value) > 0.001:
            # print("########################")
            return True
        # print("******************************")
        if self.front_value == other.front_value:
            return self.crowding_distance > other.crowding_distance
        return False

    def __eq__(self, other):
        return (
            self.front_value - other.front_value
        ) < 0.001 and self.crowding_distance == other.crowding_distance

    # def __eq__(self, other):
    #     return self._rank == other.rank and self.crowding_distance == other.crowding_distance

    # def __eq__(self, other):
    #     return all(
    #         [x == y for x, y in zip(self.front_frequency, other.front_frequency)]
    #     )

    def __repr__(self):
        return (
            f"\nObjectives:{self.objective_values},\nPolar Objectives: {self.polar_objective_values},\nChromosomes: {self.chromosome},"
            f"\nRank: {self._rank}, Crowding Distance: {self.crowding_distance}"
        )

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return (
            hash(str(self.objective_values))
            + hash(str(self.chromosome))
            + hash(self.id)
        )
