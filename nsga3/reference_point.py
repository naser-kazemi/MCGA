from emoa.utils import *


class ReferencePoint(list):
    """
    Reference Point class to represent a reference point.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.associations_count: int = 0
        self.associations: list = []

    def perpendicular_distance(self, direction) -> float:
        """
        Calculate the perpendicular distance of the member to the reference point
        :param direction: The direction of the perpendicular distance
        :return: The perpendicular distance
        """

        direction = np.array(direction) / np.linalg.norm(direction)

        k = np.dot(self, direction) / np.dot(direction, direction)
        d = np.linalg.norm(np.subtract(self, np.multiply(k, direction)))

        return d
