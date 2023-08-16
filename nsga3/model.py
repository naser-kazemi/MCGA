from itertools import chain

import numpy as np

from deap import base, tools, algorithms, creator
from deap.tools._hypervolume import hv
from emoa.utils import *
import array
import copy


class NSGA3:
    def __init__(
            self,
            problem,
            num_variables,
            num_objectives,
            num_generations,
            population_size,
            lower_bound,
            upper_bound,
            num_divisions,
            crossover_probability=0.9,
            eta_crossover=20.0,
            eta_mutation=20.0,
            log=None,
            nd="log",
            verbose=False,
    ):
        self.num_objectives = num_objectives
        self.num_generations = num_generations
        self.num_divisions = num_divisions
        self.population_size = population_size

        self.nd_sort = self.init_ndsort(nd)

        self.toolbox = self.init_toolbox(
            problem,
            num_variables,
            lower_bound,
            upper_bound,
            crossover_probability,
            eta_crossover,
            eta_mutation,
        )
        self.population = self.toolbox.population(n=population_size)
        self.current_generation = 1

        self.log = log if log is not None else []
        self.verbose = verbose
        self.stats = self.init_stats()
        self.result_pop = None
        self.logbook = None

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
            "FitnessMin", base.Fitness, weights=(-1.0,) * self.num_objectives
        )
        creator.create(
            "Individual", array.array, typecode="d", fitness=creator.FitnessMin
        )

    def init_toolbox(
            self,
            problem,
            num_variables,
            lower_bound,
            upper_bound,
            crossover_probability,
            eta_crossover,
            eta_mutation,
    ) -> base.Toolbox:

        self.create_individual_class()

        toolbox = base.Toolbox()

        toolbox.register("evaluate", problem)
        toolbox.register("select", self.select)

        toolbox.register("attr_float", uniform, lower_bound, upper_bound, num_variables)
        toolbox.register(
            "individual", tools.initIterate, creator.Individual, toolbox.attr_float
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register(
            "mate",
            tools.cxSimulatedBinaryBounded,
            low=lower_bound,
            up=upper_bound,
            eta=eta_crossover,
        )
        toolbox.register(
            "mutate",
            tools.mutPolynomialBounded,
            low=lower_bound,
            up=upper_bound,
            eta=eta_mutation,
            indpb=1.0 / num_variables,
        )

        toolbox.pop_size = self.population_size
        toolbox.max_gen = self.num_generations
        toolbox.mut_prob = 1 / num_variables
        toolbox.cross_prob = crossover_probability

        return toolbox

    def init_stats(self):
        stats = tools.Statistics()
        stats.register("pop", copy.deepcopy)
        if "fit" in self.log:
            stats.register("fit", copy.deepcopy)
        if "ndf" in self.log:
            stats.register("ndf", copy.deepcopy)
        if "ref_points" in self.log:
            stats.register("ref_points", copy.deepcopy)
        if "min" in self.log:
            stats.register("min", np.min, axis=0)
        if "max" in self.log:
            stats.register("max", np.max, axis=0)
        if "avg" in self.log:
            stats.register("avg", np.mean, axis=0)
        if "std" in self.log:
            stats.register("std", np.std, axis=0)
        return stats

    def generate_reference_points(self, scaling=None):
        """
        Generate reference points uniformly on the hyperplane intersecting
        each axis at 1. The scaling factor is used to combine multiple layers of
        reference points.
        :param scaling: The scaling factor.
        """

        nobj = self.num_objectives
        p = self.num_divisions

        def gen_refs_recursive(ref, nobj, left, total, depth):
            points = []
            if depth == nobj - 1:
                ref[depth] = left / total
                points.append(ref)
            else:
                for i in range(left + 1):
                    ref[depth] = i / total
                    points.extend(
                        gen_refs_recursive(ref.copy(), nobj, left - i, total, depth + 1)
                    )
            return points

        ref_points = np.array(gen_refs_recursive(np.zeros(nobj), nobj, p, p, 0))
        if scaling is not None:
            ref_points *= scaling
            ref_points += (1 - scaling) / nobj

        return ref_points

    @staticmethod
    def find_extreme_points(fitnesses, best_point):
        """
        Finds the individuals with extreme values for each objective function.
        :param fitnesses: The fitnesses of the population.
        :param best_point: The best point of the population.
        :return: The extreme points.
        """

        # Translate objectives
        ft = fitnesses - best_point

        # Find achievement scalarizing function (asf)
        asf = np.eye(best_point.shape[0])
        asf[asf == 0] = 1e6
        asf = np.max(ft * asf[:, np.newaxis, :], axis=2)

        # Extreme point are the fitnesses with minimal asf
        min_asf_idx = np.argmin(asf, axis=1)
        return fitnesses[min_asf_idx, :]

    @staticmethod
    def find_intercepts(extreme_points, best_point, current_worst, front_worst):
        """
        Find intercepts between the hyperplane and each axis with
        the ideal point as origin.
        """

        b = np.ones(extreme_points.shape[1])
        A = extreme_points - best_point
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            intercepts = current_worst
        else:
            if np.count_nonzero(x) != len(x):
                intercepts = front_worst
            else:
                intercepts = 1 / x

                if (
                        not np.allclose(np.dot(A, x), b)
                        or np.any(intercepts <= 1e-6)
                        or np.any((intercepts + best_point) > current_worst)
                ):
                    intercepts = front_worst

        return intercepts

    @staticmethod
    def associate(fitnesses, reference_points, best_point, intercepts):
        """
        Associates individuals to reference points and calculates niche number.
        Corresponds to Algorithm 3.
        """
        fn = (fitnesses - best_point) / (intercepts - best_point + np.finfo(float).eps)

        # Create distance matrix
        fn = np.repeat(np.expand_dims(fn, axis=1), len(reference_points), axis=1)
        norm = np.linalg.norm(reference_points, axis=1)

        distances = np.sum(fn * reference_points, axis=2) / norm.reshape(1, -1)
        distances = (
                distances[:, :, np.newaxis]
                * reference_points[np.newaxis, :, :]
                / norm[np.newaxis, :, np.newaxis]
        )
        distances = np.linalg.norm(distances - fn, axis=2)

        # Retrieve min distance niche index
        niches = np.argmin(distances, axis=1)
        distances = distances[list(range(niches.shape[0])), niches]
        return niches, distances

    @staticmethod
    def niching(individuals, k, niches, distances, niche_counts):
        selected = []
        available = np.ones(len(individuals), dtype=bool)
        while len(selected) < k:
            # Maximum number of individuals (niches) to select in that round
            n = k - len(selected)

            # Find the available niches and the minimum niche count in them
            available_niches = np.zeros(len(niche_counts), dtype=bool)
            available_niches[np.unique(niches[available])] = True
            min_count = np.min(niche_counts[available_niches])

            # Select at most n niches with the minimum count
            selected_niches = np.flatnonzero(
                np.logical_and(available_niches, niche_counts == min_count)
            )
            np.random.shuffle(selected_niches)
            selected_niches = selected_niches[:n]

            for niche in selected_niches:
                # Select from available individuals in niche
                niche_individuals = np.flatnonzero(
                    np.logical_and(niches == niche, available)
                )
                np.random.shuffle(niche_individuals)

                # If no individual in that niche, select the closest to reference
                # Else select randomly
                if niche_counts[niche] == 0:
                    sel_index = niche_individuals[
                        np.argmin(distances[niche_individuals])
                    ]
                else:
                    sel_index = niche_individuals[0]

                # Update availability, counts and selection
                available[sel_index] = False
                niche_counts[niche] += 1
                selected.append(individuals[sel_index])

        return selected

    def select(self, individuals, k):

        ref_points = self.generate_reference_points()

        pareto_fronts = self.nd_sort(individuals, k, first_front_only=False)

        fitnesses = np.array([ind.fitness.wvalues for f in pareto_fronts for ind in f])
        fitnesses *= -1

        # Get best and worst point of population, contrary to pymoo
        # we don't use memory
        best_point = np.min(fitnesses, axis=0)
        worst_point = np.max(fitnesses, axis=0)

        extreme_points = self.find_extreme_points(fitnesses, best_point)
        front_worst = np.max(fitnesses[: sum(len(f) for f in pareto_fronts), :], axis=0)
        intercepts = self.find_intercepts(
            extreme_points, best_point, worst_point, front_worst
        )
        niches, dist = self.associate(fitnesses, ref_points, best_point, intercepts)

        # Get counts per niche for individuals in all front but the last
        niche_counts = np.zeros(len(ref_points), dtype=np.int64)
        index, counts = np.unique(niches[: -len(pareto_fronts[-1])], return_counts=True)
        niche_counts[index] = counts

        # Choose individuals from all fronts but the last
        chosen = list(chain(*pareto_fronts[:-1]))

        # Use niching to select the remaining individuals
        sel_count = len(chosen)
        n = k - sel_count
        selected = self.niching(
            pareto_fronts[-1], n, niches[sel_count:], dist[sel_count:], niche_counts
        )
        chosen.extend(selected)

        self.print_stats(chosen=chosen)

        return chosen

    def print_stats(self, chosen=None):
        print("\n" + "=" * 80)
        print(f"Generation {self.current_generation}, population size: {len(chosen)}")
        self.current_generation += 1

        if not self.verbose:
            return

        if "hv" in self.log:
            print(f"HyperVolume: {self.hyper_volume(chosen)}", end="\t")

        logbook = self.stats.compile(chosen)

        for key in self.log:
            if key in logbook:
                print(f"{key}: {logbook[key]}", end="\t")

        print("\n" + "=" * 80 + "\n")

    def run(self):
        toolbox = self.toolbox
        population = toolbox.population(n=toolbox.pop_size)
        stats = self.stats
        self.result_pop, self.logbook = algorithms.eaMuPlusLambda(
            population,
            toolbox,
            mu=toolbox.pop_size,
            lambda_=toolbox.pop_size,
            cxpb=toolbox.cross_prob,
            mutpb=toolbox.mut_prob,
            ngen=toolbox.max_gen,
            stats=stats,
            verbose=False,
        )

    def metric(self, metric="hypervolume", **kwargs):
        if metric == "hypervolume":
            return self.hyper_volume(
                kwargs.get("population", self.result_pop),
                kwargs.get("ref", None),
                kwargs.get("all_gens", False),
            )
        else:
            raise ValueError("Metric not supported")

    def hyper_volume(self, population, ref=None, all_gens=False):
        def hyper_volume_util(population, ref=None):
            front = self.nd_sort(population, len(population), first_front_only=True)
            wobjs = np.array([ind.fitness.wvalues for ind in front]) * -1
            if ref is None:
                ref = np.max(wobjs, axis=0) + 1
            return hv.hypervolume(wobjs, ref)

        if all_gens:
            pops = self.logbook.select("pop")
            pops_obj = [
                np.array([ind.fitness.wvalues for ind in pop]) * -1 for pop in pops
            ]
            ref = np.max([np.max(wobjs, axis=0) for wobjs in pops_obj], axis=0) + 1
            return [hyper_volume_util(pop, ref) for pop in pops]
        else:
            return hyper_volume_util(population, ref)
