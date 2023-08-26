from itertools import chain
from operator import attrgetter

import numpy as np

from deap import base, tools, algorithms, creator
from deap.tools._hypervolume import hv
from deap.tools.emo import assignCrowdingDist

from emoa.utils import *
import array
import copy


class NSGA2:
    def __init__(
            self,
            problem,
            num_variables,
            num_objectives,
            num_generations,
            population_size,
            lower_bound,
            upper_bound,
            crossover_probability=0.9,
            eta_crossover=20.0,
            eta_mutation=20.0,
            log=None,
            nd="log",
            verbose=False,
    ):
        self.num_variables = num_variables
        self.num_objectives = num_objectives
        self.num_generations = num_generations
        self.population_size = population_size
        self.cross_prob = crossover_probability

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
            "FitnessMin",
            base.Fitness,
            weights=(-1.0,) * self.num_objectives,
            crowding_dist=0.0,
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

        toolbox.register("attr_float", uniform, lower_bound, upper_bound, num_variables)
        toolbox.register(
            "individual", tools.initIterate, creator.Individual, toolbox.attr_float
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", problem)
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=lower_bound, up=upper_bound, eta=eta_crossover)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=lower_bound, up=upper_bound, eta=eta_mutation,
                         indpb=1.0 / num_variables)
        toolbox.register("select", self.select)

        return toolbox

    def init_stats(self):
        stats = tools.Statistics()
        # stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("pop", copy.deepcopy)
        if "fit" in self.log:
            stats.register("fit", copy.deepcopy)
        if "min" in self.log:
            stats.register("min", np.min, axis=0)
        if "max" in self.log:
            stats.register("max", np.max, axis=0)
        if "avg" in self.log:
            stats.register("avg", np.mean, axis=0)
        if "std" in self.log:
            stats.register("std", np.std, axis=0)
        return stats

    def select(self, individuals, k):
        pareto_fronts = self.nd_sort(individuals, k)

        for front in pareto_fronts:
            assignCrowdingDist(front)

        chosen = list(chain(*pareto_fronts[:-1]))
        k = k - len(chosen)
        if k > 0:
            sorted_front = sorted(
                pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True
            )
            chosen.extend(sorted_front[:k])

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
        population = toolbox.population(n=self.population_size)
        stats = self.stats
        self.result_pop, self.logbook = algorithms.eaMuPlusLambda(
            population,
            toolbox,
            mu=self.population_size,
            lambda_=self.population_size,
            cxpb=self.cross_prob,
            mutpb=1.0 / self.num_variables,
            ngen=self.num_generations,
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
            # front = self.nd_sort(population, len(population), first_front_only=True)
            # objs = np.array([ind.fitness.values for ind in population]) * -1
            # objs = np.array([ind.fitness.wvalues for ind in front]) * -1
            objs = np.array(population)
            if ref is None:
                ref = np.max(objs, axis=0) + 1
            return hv.hypervolume(objs, ref)

        if all_gens:
            pops = self.logbook.select("pop")
            # pops_obj = [
            #     np.array([ind.fitness.wvalues for ind in pop]) * -1 for pop in pops
            # ]
            # ref = np.max([np.max(objs, axis=0) for objs in pops_obj], axis=0) + 1
            return [hyper_volume_util(pop, ref) for pop in pops]
        else:
            return hyper_volume_util(population, ref)
