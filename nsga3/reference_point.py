class ReferencePoint(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.niche_count = 0
        self.associated_individuals = []

    def associate_individual(self, individual):
        self.associated_individuals.append(individual)
        self.niche_count += 1

    def remove_associated_individual(self, individual):
        self.associated_individuals.remove(individual)
        self.niche_count += 1

    def __repr__(self) -> str:
        return (
            f"ReferencePoint({super().__repr__()}), "
            f"niche_count={self.niche_count}, "
            f"associated_members={self.associated_individuals}"
        )

    def __str__(self) -> str:
        return self.__repr__()
