import array
import copy
from itertools import product

import numpy as np
from scipy.spatial import Delaunay

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
        super().__init__(
            lambda x: np.zeros(num_objectives),
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

        self.plot_dir = os.path.join(
            "printer_plots", exploration_params.model, exploration_params.name
        )
        self.all_design_space = None
        self.all_performance_space = None
        self.all_xyz_colors = None
        self.all_samples = []
        self.areas = np.zeros(self.num_generations + 1)
        self.nd_sort = self.init_ndsort(exploration_params.nd)
        self.fig = plt.figure(figsize=(10, 10))

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
            idx=0,
            front_freq=np.zeros(self.population_size * 2, dtype=np.float64),
            performance_space=np.zeros(3, dtype=np.float64),
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
        # zero = np.zeros(len(individuals))
        # obj_scores = np.column_stack((obj1, zero, zero))

        # a_star = performance_space[:, 1]
        # b_star = performance_space[:, 2]
        # obj_scores = np.column_stack((a_star, b_star))

        obj_scores = np.column_stack((obj1, obj2, obj3))

        for i in range(len(individuals)):
            individuals[i].fitness.values = obj_scores[i]
            individuals[i].fitness.performance_space = performance_space[i]

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

    def run(self):
        population, xyz_colors_p0, points_ps_p0 = self.init_population()

        # compute and visualize gamut area of the test samples
        p0_area = compute_area(xyz_colors_p0)
        self.areas[0] = p0_area

        print("Gamut area of initial collection is %.6f" % p0_area)
        visualization.save_lab_gamut(
            points_ps_p0,
            self.plot_dir,
            "gamut_a_initial",
            "Initial gamut (area=%.3f)" % p0_area,
        )

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + self.stats.fields

        record = self.stats.compile(population)
        del record["pop"]

        logbook.record(gen=0, **record)
        if self.verbose:
            print(logbook.stream)

        for g in range(1, self.num_generations + 1):
            self.evaluate(population)
            offspring = varOr(
                population,
                self.toolbox,
                self.population_size,
                self.crossover_probability,
                1.0 / self.num_variables,
            )

            # combine offspring and population
            chosen = self.select(population + offspring, self.population_size)

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

            self.all_performance_space = np.vstack(
                (self.all_performance_space, points_ps)
            )
            self.all_xyz_colors = np.vstack((self.all_xyz_colors, xyz_colors_qi))
            # print(self.all_performance_space)

            # compute and visualize gamut area of the test samples
            current_area = compute_area(self.all_xyz_colors)
            self.areas[g] = current_area
            print("Gamut area of current collection is %.6f" % current_area)
            visualization.save_lab_gamut(
                self.all_performance_space,
                self.plot_dir,
                "gamut_after_iter_%d" % g,
                "Gamut after iteration %d (area=%.3f)" % (g, current_area),
            )

            record = self.stats.compile(population)
            del record["pop"]
            logbook.record(gen=g, **record)
            if self.verbose:
                print(logbook.stream)

            self.current_generation += 1

        self.result_pop = population
        self.logbook = logbook

    def select(self, individuals, k):
        self.evaluate(individuals)
        ranks, scores, sorted_ids, point_fronts = self.mc_sort(individuals)
        chosen = [individuals[i] for i in sorted_ids[:k]]
        return chosen

    def mc_sort(self, individuals):
        mc_samples = 0
        slice_radius = 100

        num_individuals = len(individuals)
        for i in range(num_individuals):
            individuals[i].fitness.idx = i

        point_fronts = np.zeros((num_individuals, num_individuals))
        norm_point_fronts = np.zeros((num_individuals, num_individuals))

        min_mc_samples = 1
        max_mc_samples = 1000
        avg_diff = np.inf

        while (
            (avg_diff > self.front_frequency_threshold) or (mc_samples < min_mc_samples)
        ) and (mc_samples < max_mc_samples):
            cr = np.random.rand()
            ora = np.random.rand()

            start_angle = self.polar_offset_limit[0] + ora * (
                self.polar_offset_limit[1] - self.polar_offset_limit[0]
            )
            slice_count = self.num_max_sectors[0] + round(
                cr * (self.num_max_sectors[1] - self.num_max_sectors[0])
            )
            rad_per_slice = (
                self.polar_offset_limit[1] - self.polar_offset_limit[0]
            ) / slice_count
            prev_point_fronts = point_fronts.copy()
            for s in range(slice_count):
                slx, sly = vector_to_cartesian(
                    slice_radius, np.array([start_angle + s * rad_per_slice])
                )
                srx, sry = vector_to_cartesian(
                    slice_radius, np.array([start_angle + (s + 1) * rad_per_slice])
                )
                poly = np.array([[0, 0], [slx, sly], [srx, sry], [0, 0]])

                # point = np.column_stack([individuals.fitness.values[:, 1], individuals.fitness.values[:, 2]])
                point = np.array(
                    [ind.fitness.performance_space[1:] for ind in individuals]
                )
                # point = np.array([ind.fitness.values[1:] for ind in individuals])

                in_mask = Delaunay(poly).find_simplex(point)
                slice_points_ids = np.where(in_mask != -1)[0]

                if len(slice_points_ids) != 0:
                    curr_inds = [individuals[i] for i in slice_points_ids]
                    fronts = self.nd_sort(curr_inds, len(curr_inds))

                    for f in range(len(fronts)):
                        p_ids = np.array([ind.fitness.idx for ind in fronts[f]])
                        point_fronts[p_ids, f] = point_fronts[p_ids, f] + 1

            # new_norm_point_fronts = point_fronts / point_fronts.sum(axis=1, keepdims=True)
            # print(point_fronts)
            new_norm_point_fronts = point_fronts / np.linalg.norm(
                point_fronts, ord="fro"
            )
            if mc_samples > 0:
                prev_point_fronts = prev_point_fronts / np.linalg.norm(
                    prev_point_fronts, ord="fro"
                )
            diffs = np.linalg.norm(new_norm_point_fronts - prev_point_fronts, ord="fro")
            # avg_diff = np.mean(diffs)
            avg_diff = diffs
            norm_point_fronts = new_norm_point_fronts

            mc_samples += 1

        print(f"Computed fronts (#{mc_samples} hue wheel samplings, {avg_diff})")
        sorted_ids = np.lexsort(-point_fronts.T[::-1])
        ranks = np.zeros(len(individuals), dtype=np.int64)
        ranks[sorted_ids] = np.arange(len(individuals))
        scores = (len(individuals) - ranks) / len(individuals)
        sorted_ids = ranks.argsort()

        scores = scores / np.sum(scores)

        return ranks, scores, sorted_ids, point_fronts
