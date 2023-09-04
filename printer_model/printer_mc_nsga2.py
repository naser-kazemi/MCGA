import array
import copy
from itertools import product

from nsga2 import NSGA2
from .printer_utils import *
from . import exploration_params, visualization
from .printer_objectives import *


class PrinterMCNSGA2(NSGA2):
    def __init__(self):
        num_variables = 3
        num_objectives = 3
        limits_ds = np.array(exploration_params.limits_ds)
        lower_bound = limits_ds[:, 0].tolist()
        upper_bound = limits_ds[:, 1].tolist()
        population_size = exploration_params.pop_size
        super().__init__(lambda x: np.zeros(num_objectives),
                         num_variables,
                         num_objectives,
                         num_generations=exploration_params.iterations,
                         population_size=population_size,
                         lower_bound=lower_bound,
                         upper_bound=upper_bound,
                         crossover_probability=exploration_params.crossover_probability,
                         eta_crossover=exploration_params.eta_crossover,
                         eta_mutation=exploration_params.eta_mutation,
                         log=exploration_params.log,
                         nd=exploration_params.nd,
                         verbose=exploration_params.verbose,
                         )
        self.num_max_sectors = exploration_params.num_max_sectors
        self.polar_offset_limit = exploration_params.polar_offset_limit
        self.front_frequency_threshold = exploration_params.front_frequency_threshold
        self.scale = 1.0

        self.plot_dir = os.path.join("printer_plots", exploration_params.model, exploration_params.name)
        self.all_design_space = None
        self.all_performance_space = None
        self.all_xyz_colors = None
        self.all_samples = []
        self.areas = np.zeros(self.num_generations + 1)
        self.nd_sort = self.init_ndsort(exploration_params.nd)

    def init_ndsort(self, nd):
        if nd == "standard":
            return tools.sortNondominated
        elif nd == "log":
            return tools.sortLogNondominated
        else:
            raise Exception(
                "The choice of non-dominated sorting "
                "method '{0}' is invalid.".format(nd)
            )

    def create_individual_class(self):
        creator.create(
            "FitnessMin",
            base.Fitness,
            weights=(1.0,) * self.num_objectives,
            polar_coords=np.zeros(self.num_objectives, dtype=np.float64),
            front_freq=np.zeros(self.population_size * 2, dtype=np.float64),
        )
        creator.create(
            "Individual", array.array, typecode="d", fitness=creator.FitnessMin
        )

    def evaluate(self, individuals):
        design_space = np.array(individuals)
        [_, xyz_colors, performance_space] = predict_printer_colors(
            design_space,
            exploration_params.ng_primary_reflectances,
            exploration_params.white_ciexyz,
            exploration_params.d65_illuminant,
            exploration_params.xbar,
            exploration_params.ybar,
            exploration_params.zbar,
        )

        obj1 = chromaticity_obj(performance_space[:, 1:3])
        obj2 = diversity_mult_obj(
            design_space, exploration_params.k_n, exploration_params.limits_ds
        )
        obj3 = diversity_mult_obj(
            performance_space[:, 1:3],
            exploration_params.k_n,
            exploration_params.limits_ps[1:3],
        )
        obj_scores = np.column_stack((obj1, obj2, obj3))

        for i in range(len(individuals)):
            individuals[i].fitness.values = obj_scores[i]

    def init_population(self):
        population = self.toolbox.population(n=self.population_size)
        points_ds_p0 = get_random_parameters(
            exploration_params.limits_ds,
            exploration_params.prec_facs,
            exploration_params.cont_ds,
            exploration_params.pop_size,
        )

        [_, xyz_colors_p0, points_ps_p0] = predict_printer_colors(
            points_ds_p0,
            exploration_params.ng_primary_reflectances,
            exploration_params.white_ciexyz,
            exploration_params.d65_illuminant,
            exploration_params.xbar,
            exploration_params.ybar,
            exploration_params.zbar,
        )

        self.all_design_space = points_ds_p0.copy()
        self.all_performance_space = points_ps_p0.copy()
        self.all_xyz_colors = xyz_colors_p0.copy()

        for i in range(self.population_size):
            for v in range(self.num_variables):
                population[i][v] = points_ds_p0[i][v]

        return population, xyz_colors_p0, points_ps_p0

    # def validate_individuals(self, population):
    #     invalid_ind = [ind for ind in population if not ind.fitness.valid]
    #     self.evaluate(invalid_ind)
    #     return invalid_ind

    def run(self):
        population, xyz_colors_p0, points_ps_p0 = self.init_population()

        # compute and visualize gamut area of the test samples
        p0_area = compute_area(xyz_colors_p0)
        self.areas[0] = p0_area

        print("Gamut area of initial collection is %.6f" % p0_area)
        visualization.save_lab_gamut(
            points_ps_p0, self.plot_dir, "gamut_a_initial", "Initial gamut (area=%.3f)" % p0_area
        )

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + self.stats.fields

        record = self.stats.compile(population)
        del record['pop']

        logbook.record(gen=0, **record)
        if self.verbose:
            print(logbook.stream)

        for g in range(1, self.num_generations + 1):
            self.evaluate(population)
            offspring = varOr(population, self.toolbox, self.population_size, self.crossover_probability,
                              1.0 / self.num_variables)

            # combine offspring and population
            population = population + offspring
            self.evaluate(population)
            chosen = self.select(population, self.population_size)

            points_ds = np.array(chosen)
            [_, xyz_colors_qi, points_ps] = predict_printer_colors(
                points_ds,
                exploration_params.ng_primary_reflectances,
                exploration_params.white_ciexyz,
                exploration_params.d65_illuminant,
                exploration_params.xbar,
                exploration_params.ybar,
                exploration_params.zbar,
            )
            population = chosen

            self.all_performance_space = np.vstack((self.all_performance_space, points_ps))
            self.all_xyz_colors = np.vstack((self.all_xyz_colors, xyz_colors_qi))
            # print(self.all_performance_space)

            # compute and visualize gamut area of the test samples
            current_area = compute_area(self.all_xyz_colors)
            self.areas[g] = current_area
            print("Gamut area of current collection is %.6f" % current_area)
            visualization.save_lab_gamut(
                self.all_performance_space, self.plot_dir, "gamut_after_iter_%d" % g,
                                                           "Gamut after iteration %d (area=%.3f)" % (g, current_area)
            )

            record = self.stats.compile(population)
            del record['pop']
            logbook.record(gen=g, **record)
            if self.verbose:
                print(logbook.stream)

            self.current_generation += 1

        self.result_pop = population
        self.logbook = logbook

    def divide_planes(self):
        """
        Create sectors in polar space,
        each sector is a tuple of (start, end) angles in n-dimensional polar space
        where n is the number of objectives
        :return: The sectors
        """

        # create a list of sectors
        sectors_points = []

        # divide the 2 * pi radians into num_sectors sectors evenly
        for i in range(self.num_objectives - 1):
            sector = np.linspace(0, 2 * np.pi, self.num_max_sectors + 1)
            sectors_points.append(sector)

        sectors = []
        # now create the (start, end) tuples for each sector
        for i in range(self.num_objectives - 1):
            sectors.append(
                [
                    (sectors_points[i][j], sectors_points[i][j + 1])
                    for j in range(len(sectors_points[i]) - 1)
                ]
            )

        # now rotate the sectors to create the offset
        offset = random.uniform(0, self.polar_offset_limit)
        for i in range(self.num_objectives - 1):
            sectors[i] = [(x + offset, y + offset) for x, y in sectors[i]]

        return sectors

    @classmethod
    def create_sectors(cls, plane_sectors):
        """
        Create sectors in polar space,
        each sector is list of a tuples of (start, end) angles in n-dimensional polar space
        where n is the number of objectives
        The sectors are cartesian product of the plane sectors
        :return: The sectors
        """

        # create the cartesian product of the plane sectors
        sectors = np.array(list(product(*plane_sectors)))

        return sectors

    @staticmethod
    def convert_to_polar(individuals):
        """
        Convert the individuals in the population to polar coordinates
        :return: None
        """
        for ind in individuals:
            ind.fitness.polar_coords = vector_to_polar(ind.fitness.values)[1]

    @staticmethod
    def is_in_sector(sector, individual):
        is_in_bounds = True
        for x, (start, end) in zip(individual.fitness.polar_coords[1:], sector):
            if end <= 2 * np.pi:
                in_this_sector = start <= x < end
            else:
                in_this_sector = start <= x < 2 * np.pi or 0 <= x < end - 2 * np.pi
            is_in_bounds = is_in_bounds and in_this_sector

        return is_in_bounds

    def slice_polar_space(self, individuals):
        """
        Slice the polar space into sectors and assign each member of the population to a sector
        :param individuals: The population to slice
        :return: The sliced population in polar space
        """

        plane_sectors = self.divide_planes()
        sectors = self.create_sectors(plane_sectors)

        sliced_population = [[] for _ in range(len(sectors))]
        # divide the population into sectors
        for ind in individuals:
            for i in range(len(sectors)):
                if self.is_in_sector(sectors[i], ind):
                    sliced_population[i].append(ind)
                    break
            else:
                print(f"Error: Member {ind.fitness.polar_coords[1:]} not in any sector")

        sliced_population = [slc for slc in sliced_population if len(slc) > 0]

        return sliced_population

    def mc_nds(self, sliced_population) -> None:
        """
        Monte Carlo Non-Dominated Sorting
        Apply non-dominated sorting on the sliced population and assign ranks to the members
        :param sliced_population: The sliced population
        """

        for population in sliced_population:
            fronts = self.nd_sort(population, len(population))
            for i, front in enumerate(fronts):
                for ind in front:
                    ind.fitness.front_freq[i] += 1

    def monte_carlo_step(self, individuals):

        sliced_population = self.slice_polar_space(individuals)
        self.mc_nds(sliced_population)

    @staticmethod
    def compute_front_frequency_diff(individuals, cached_individuals):
        """
        Compute the difference between the front frequencies of the individuals
        :param individuals: The individuals to compare
        :param cached_individuals: The cached individuals to compare
        """

        a = np.array([ind.fitness.front_freq for ind in individuals])
        b = np.array([ind.fitness.front_freq for ind in cached_individuals])

        a = a / np.linalg.norm(a, ord="fro")
        b = b / np.linalg.norm(b, ord="fro")

        return np.linalg.norm(a - b, ord="fro")

    @staticmethod
    def front_value(individual):
        # value = 0.0
        # c = 1.0
        # for ff in individual.fitness.front_freq:
        #     value += ff * c
        #     c *= 0.8
        # return value

        return " ".join([str(x) for x in individual.fitness.front_freq])

    def mc_select(self, individuals):
        self.convert_to_polar(individuals)
        cached_individuals = copy.deepcopy(individuals)
        self.monte_carlo_step(individuals)

        while (
                self.compute_front_frequency_diff(individuals, cached_individuals)
                > self.front_frequency_threshold
        ):
            cached_individuals = copy.deepcopy(individuals)
            self.monte_carlo_step(individuals)

        sorted_individuals = sorted(individuals, key=self.front_value, reverse=True)

        return sorted_individuals

    def select(self, individuals, k):
        """
        Select the individuals to survive to the next generation
        :param individuals: The individuals to select from
        :param k: The number of individuals to select
        :return: The selected individuals
        """

        for ind in individuals:
            ind.fitness.values = tuple([x * self.scale for x in ind.fitness.values])

        mc_sorted = self.mc_select(individuals)
        chosen = mc_sorted[: self.population_size]

        print("Chosen individuals:")
        for ind in chosen:
            print(ind.fitness.values)

        for ind in individuals:
            ind.fitness.values = tuple([x / self.scale for x in ind.fitness.values])

        # self.print_stats(chosen=chosen)

        return chosen
