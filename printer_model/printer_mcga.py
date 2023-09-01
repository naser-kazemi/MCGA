import numpy as np
from deap import algorithms, tools

from mcga import MCNSGA3
from .printer_utils import *
from . import exploration_params, visualization
from .printer_objectives import *


class PrinterMCNSGA3(MCNSGA3):
    def __init__(self):
        num_variables = 3
        num_objectives = 3
        limits_ds = np.array(exploration_params.limits_ds)
        lower_bound = limits_ds[:, 0]
        upper_bound = limits_ds[:, 1]
        population_size = exploration_params.pop_size
        super().__init__(lambda x: np.zeros(num_objectives),
                         num_variables,
                         num_objectives,
                         num_generations=exploration_params.iterations,
                         population_size=population_size,
                         lower_bound=lower_bound,
                         upper_bound=upper_bound,
                         num_divisions=exploration_params.num_divisions,
                         crossover_probability=exploration_params.crossover_probability,
                         eta_crossover=exploration_params.eta_crossover,
                         eta_mutation=exploration_params.eta_mutation,
                         num_max_sectors=exploration_params.num_max_sectors,
                         front_frequency_threshold=exploration_params.front_frequency_threshold,
                         polar_offset_limit=exploration_params.polar_offset_limit,
                         monte_carlo_frequency=exploration_params.monte_carlo_frequency,
                         log=exploration_params.log,
                         nd=exploration_params.nd,
                         verbose=exploration_params.verbose,
                         )
        self.plot_dir = os.path.join("printer_plots", exploration_params.name)
        self.all_design_space = None
        self.all_performance_space = None
        self.all_xyz_colors = None
        self.all_samples = []
        self.areas = np.zeros(self.num_generations + 1)

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
        ) * 0.001

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

    def validate_individuals(self, population):
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        return invalid_ind

    def run(self):
        population, xyz_colors_p0, points_ps_p0 = self.init_population()

        self.result_pop, self.logbook = algorithms.eaMuPlusLambda(
            population,
            self.toolbox,
            mu=self.population_size,
            lambda_=self.population_size,
            cxpb=self.crossover_probability,
            mutpb=min(1.0 / self.num_variables, 0.2),
            ngen=self.num_generations,
            stats=self.stats,
            verbose=False,
        )

        # compute and visualize gamut area of the test samples
        p0_area = compute_area(xyz_colors_p0)
        self.areas[0] = p0_area

        print("Gamut area of initial collection is %.6f" % p0_area)
        visualization.save_lab_gamut(
            points_ps_p0, self.plot_dir, "gamut_a_initial", "Initial gamut (area=%.3f)" % p0_area
        )

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + self.stats.fields

        invalid_ind = self.validate_individuals(population)

        record = self.stats.compile(population)
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if self.verbose:
            print(logbook.stream)

        points_ds_pi = self.all_design_space
        points_ps_pi = self.all_performance_space

        for g in range(1, self.num_generations + 1):
            self.evaluate(population)

            offspring = varOr(population, self.toolbox, self.population_size, self.crossover_probability,
                              1.0 / self.num_variables)

            # combine offspring and population
            population = population + offspring

            invalid_ind = self.validate_individuals(population)
            chosen = self.select(population, self.population_size)

            points_ds_qi = np.array(chosen)
            [_, xyz_colors_qi, points_ps_qi] = predict_printer_colors(
                points_ds_qi,
                exploration_params.ng_primary_reflectances,
                exploration_params.white_ciexyz,
                exploration_params.d65_illuminant,
                exploration_params.xbar,
                exploration_params.ybar,
                exploration_params.zbar,
            )

            self.all_performance_space = np.vstack((self.all_performance_space, points_ps_qi))
            self.all_xyz_colors = np.vstack((self.all_xyz_colors, xyz_colors_qi))

            # compute and visualize gamut area of the test samples
            current_area = compute_area(self.all_xyz_colors)

            print("Gamut area of current collection is %.6f" % current_area)
            visualization.save_lab_gamut(
                self.all_performance_space, self.plot_dir, "gamut_a_current", "Current gamut (area=%.3f)" % current_area
            )

            points_ds_ri = np.vstack((points_ds_pi, points_ds_qi))
            points_ps_ri = np.vstack((points_ps_pi, points_ps_qi))
            population_ri = self.toolbox.population(n=len(points_ds_ri))
            for i in range(len(points_ds_ri)):
                for v in range(self.num_variables):
                    population_ri[i][v] = points_ds_ri[i][v]

            
