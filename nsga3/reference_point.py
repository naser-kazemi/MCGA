from emoa.utils import *
from emoa import Member, Population


class ReferencePoint(list):
    """
    Reference Point class to represent a reference point.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.associations_count: int = 0
        self.associations: list[Member] = []

    def perpendicular_distance(self, direction):
        """
        Calculate the perpendicular distance of the member to the reference point
        :param direction: The direction of the perpendicular distance
        :return: The perpendicular distance
        """

        direction = np.array(direction) / np.linalg.norm(direction)

        k = np.dot(self, direction) / np.dot(direction, direction)
        d = np.linalg.norm(np.subtract(self, np.multiply(k, direction)))
