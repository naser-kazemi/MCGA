import random
from .moop import MOOP
from .member import Member
from .population import Population
import matplotlib.pyplot as plt
from .utils import np, generate_color


class NSGA2:
    """
    NSGA-II algorithm implementation
    The algorithm is based on the paper:
    A fast and elitist multiobjective genetic algorithm: NSGA-II
    by Kalyanmoy Deb, Samir Agrawal, Amrit Pratap, and T Meyarivan
    It will contain the following attributes:
        - moop: The multi-objective optimization problem
        - num_generation: The number of generations
        - population_size: The size of the population
        - population: The population
        - offsprings: The offsprings
        - crossover_probability: The crossover probability
        - mutation_probability: The mutation probability
        - tournament_size: The tournament size
        - eta_crossover: The eta crossover
        - eta_mutation: The eta mutation
    """

    def __init__(
            self,
            moop: MOOP,
            num_generation: int,
            population_size: int,
            crossover_probability: float = 0.9,
            tournament_size: int = 2,
            eta_crossover: float = 1.0,
            eta_mutation: float = 1.0,
    ):
        self.moop = moop
        self.num_generation = num_generation
        self.population_size = population_size
        self.population = self.init_population()
        self.offsprings: Population = Population()
        self.crossover_probability = crossover_probability
        self.mutation_probability = 1 / (moop.num_objectives + 5)
        self.tournament_size = tournament_size
        self.eta_crossover = eta_crossover
        self.eta_mutation = eta_mutation

    def create_member(self) -> Member:
        """
        Create a member of the population
        :return: The created member
        """
        chromosome = self.moop.generate_chromosome()
        objective_values = self.moop.evaluate(chromosome)
        member = Member(chromosome, objective_values)
        member.to_polar()
        member.front_frequency = [0 for _ in range(self.population_size * 2)]
        return member

    def evaluate_population(self, population: Population = None) -> None:
        """
        Evaluate the population
        :param population: The population to evaluate
        :return: None
        """
        if population is None:
            population = self.population
        for member in population:
            member.objective_values = self.moop.evaluate(member.chromosome)
            member.to_polar()

    def init_population(self) -> Population:
        """
        Initialize the population
        :return: The initialized population
        """
        population = Population()
        for i in range(self.population_size):
            population.append(self.create_member())
        return population

    @classmethod
    def fast_non_dominated_sort(cls, population: Population) -> list[list]:
        """
        Fast Non-Dominated Sorting
        :param population: The population to sort
        :return: A list of fronts
        """

        dominated_members = {member: [] for member in population}
        fronts = [[]]
        for p in population:
            for q in population:
                if p.dominates(q):
                    dominated_members[p].append(q)
                    q.dominated_by_count += 1

        for p in population:
            if p.dominated_by_count == 0:
                p.rank = 1
                fronts[0].append(p)

        i = 1
        while fronts[-1]:
            next_front = []
            for p in fronts[-1]:
                for q in dominated_members[p]:
                    q.dominated_by_count -= 1
                    if q.dominated_by_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts

    @classmethod
    def normalize_values(cls, values: list[float]):
        v_max = max(values)
        v_min = min(values)
        scale = v_max - v_min
        if abs(scale) < 1e-10 or scale == float("inf"):
            return [0 for _ in values]
        return [(v - v_min) / scale for v in values]

    @classmethod
    def compute_crowding_distance(
            cls, front: list[Member], num_objectives: int
    ) -> None:
        """
        Compute the crowding distance for a front
        :param front: The front to compute the crowding distance for
        :param num_objectives: the number of objectives for member of fronts
        :return: The front with the crowding distance computed
        """

        n = len(front)
        if n == 0:
            return

        for member in front:
            member.crowding_distance = 0

        for m in range(num_objectives):
            front.sort(key=lambda x: x.objective_values[m])
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")
            norm_values = cls.normalize_values(
                [member.objective_values[m] for member in front]
            )
            for i in range(1, n - 1):
                front[i].crowding_distance += norm_values[i + 1] - norm_values[i - 1]

    def mutate(self, member: Member) -> Member:
        """
        Perform mutation on a member with a probability of self.mutation_probability
        :param member: The member to mutate
        :return: The mutated member
        """
        for i in range(self.moop.num_variables):
            if random.random() < self.mutation_probability:
                member.chromosome[i] += random.uniform(
                    -0.1 * self.eta_mutation, 0.1 * self.eta_mutation
                )
            if member.chromosome[i] < self.moop.lower_bounds[i]:
                member.chromosome[i] = self.moop.lower_bounds[i]
            elif member.chromosome[i] > self.moop.upper_bounds[i]:
                member.chromosome[i] = self.moop.upper_bounds[i]
        return member

    def crossover(self, parent1: Member, parent2: Member) -> tuple[Member, Member]:
        """
        Crossover two parents with a probability of self.crossover_probability
        :param parent1: The first parent
        :param parent2: The second parent
        :return: The two children
        """

        if random.random() > self.crossover_probability:
            return parent1, parent2

        crossover_point = random.randint(0, self.moop.num_variables - 1)
        child1 = parent1.copy()
        child2 = parent2.copy()

        child1.chromosome[crossover_point:] = parent2.chromosome[crossover_point:]
        child2.chromosome[crossover_point:] = parent1.chromosome[crossover_point:]

        return child1, child2

    def tournament(self):
        member1: Member = random.choice(self.population.population)
        member2: Member = random.choice(self.population.population)
        if member1 > member2:
            return member1
        return member2

    def make_new_population(self) -> Population:
        """
        Make a new population from the current population
        :return: The offsprings
        """
        offsprings = Population()
        while offsprings.size < self.population_size:
            parent1, parent2 = self.tournament(), self.tournament()
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            offsprings += [child1, child2]
        return offsprings

    def run_generation(self) -> None:
        """
        Run the algorithm for one generation
        """

        self.offsprings = self.make_new_population()
        self.evaluate_population(self.offsprings)

        r = self.population + self.offsprings
        fronts = self.fast_non_dominated_sort(r)
        for front in fronts:
            self.compute_crowding_distance(front, self.moop.num_objectives)

        next_population = Population()

        i = 0
        while i < len(fronts) and next_population.size < self.population_size:
            front = fronts[i]
            front = sorted(front, key=lambda x: x.crowding_distance, reverse=True)
            if next_population.size + len(front) <= self.population_size:
                next_population += front
            else:
                next_population += front[: self.population_size - next_population.size]
            i += 1

        self.population = next_population

    def run(self) -> None:
        """
        Run the algorithm for a given number of generations
        """
        self.plot_population_frame(0, f"gif_images/generation_{0}.png")
        fronts = self.fast_non_dominated_sort(self.population)
        for front in fronts:
            self.compute_crowding_distance(front, self.moop.num_objectives)
        for i in range(self.num_generation):
            self.run_generation()
            print(f"Generation {i + 1} done")
            # create a gif of the evolution of the population
            self.plot_population_frame(i + 1, f"gif_images/generation_{i + 1}.png")

    def plot_population_frame(self, generation, filename: str) -> None:
        """
        Plot the population and save the plot to a file as a frame for a gif
        :param generation: The generation number
        :param filename: The name of the file to save the plot to
        """
        dim = self.moop.num_objectives

        fig = plt.figure(figsize=(6, 6))

        objective_values = np.array(
            [member.objective_values for member in self.population]
        )

        if dim == 2:
            plt.scatter(
                self.moop.pareto_front[:, 0],
                self.moop.pareto_front[:, 1],
                color="red",
                s=10,
            )
            plt.scatter(
                objective_values[:, 0],
                objective_values[:, 1],
                color="blue",
                s=10,
                alpha=0.7,
            )
        elif dim == 3:
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                self.moop.pareto_front[:, 0],
                self.moop.pareto_front[:, 1],
                self.moop.pareto_front[:, 2],
                color="red",
                s=10,
            )
            ax.scatter(
                objective_values[:, 0],
                objective_values[:, 1],
                objective_values[:, 2],
                color="blue",
                s=10,
                alpha=0.7,
            )
        else:
            raise Exception("Cannot plot more than 3 dimensions")

        plt.xlabel("Objective 1")
        plt.ylabel("Objective 2")
        plt.title(f"Generation {generation}")
        plt.savefig(filename)
        plt.close(fig)

    def evaluate_distance_metric(self):
        """
        Evaluate the distance of the solutions in the population to the actual Pareto front
        """

        # compute the distance of each solution to the closest solution in the Pareto front
        distances = np.zeros(self.population.size)
        population_objective_values = np.array(
            [member.objective_values for member in self.population]
        )
        for i in range(self.population.size):
            distances[i] = np.min(
                np.linalg.norm(
                    population_objective_values[i] - self.moop.pareto_front, axis=1
                )
            )

        # return the average distance and the standard deviation
        return np.mean(distances), np.std(distances)

    def evaluate_diversity_metric(self):
        """
        Evaluate the diversity of the solutions in the population
        """

        front = np.array(self.fast_non_dominated_sort(self.population)[0])
        front = np.array([member.objective_values for member in front])
        front = front[np.lexsort((front[:, 1], -front[:, 0]))]

        # compute the distance between consecutive solutions
        distances = np.linalg.norm(front[1:] - front[:-1], axis=1)
        avg_distance = np.mean(distances)

        # get the extreme solutions in the pareto front
        f_extr, l_extr = self.moop.pareto_front[0], self.moop.pareto_front[-1]

        # compute the distance of the extreme solutions to the front
        dl, df = np.linalg.norm(front[0] - l_extr), np.linalg.norm(front[-1] - f_extr)

        # compute the diversity of the population
        diversity = (df + dl + np.sum(np.abs(distances - avg_distance))) / (
                df + dl + (self.population_size - 1) * avg_distance
        )

        return diversity

    def plot_population(self, ax, population: Population = None) -> None:
        if population is None:
            population = self.population

        dim = self.moop.num_objectives

        objective_values = np.array([member.polar_objective_values for member in population])

        if dim == 2:
            ax.scatter(objective_values[:, 1], objective_values[:, 0], color=generate_color(), s=10, alpha=0.7)
        elif dim == 3:
            X = objective_values[:, 0] * np.sin(objective_values[:, 2]) * np.cos(objective_values[:, 1])
            Y = objective_values[:, 0] * np.sin(objective_values[:, 2]) * np.sin(objective_values[:, 1])
            Z = objective_values[:, 0] * np.cos(objective_values[:, 2])
            ax.scatter(X, Y, Z, color=generate_color(), s=10, alpha=0.7)
