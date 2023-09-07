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
        self.crossover_probability = crossover_probability
        self.nd = nd
        self.problem = problem

        self.current_generation = 1
        self.log = log if log is not None else []
        self.verbose = verbose

        self.stats = None
        self.toolbox = None
        self.result_pop = None
        self.logbook = None
        self.create_model(
            problem,
            num_variables,
            population_size,
            lower_bound,
            upper_bound,
            crossover_probability,
            eta_crossover,
            eta_mutation,
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

    def create_model(
            self,
            problem,
            num_variables,
            population_size,
            lower_bound,
            upper_bound,
            crossover_probability,
            eta_crossover,
            eta_mutation,
    ):
        self.create_individual_class()

        toolbox = base.Toolbox()
        toolbox.register("attr_float", uniform, lower_bound, upper_bound, num_variables)
        toolbox.register(
            "individual", tools.initIterate, creator.Individual, toolbox.attr_float
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", problem)
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
            indpb=min(1.0 / num_variables, 0.2),
        )
        toolbox.register("select", self.select)

        self.toolbox = toolbox

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("pop", copy.deepcopy)

    def select(self, individuals, k):
        chosen = tools.selNSGA2(individuals, k, nd=self.nd)
        self.print_stats(chosen=chosen)
        return chosen

    def run(self):
        pop = self.toolbox.population(n=self.population_size)
        self.result_pop, self.logbook = algorithms.eaMuPlusLambda(
            pop,
            self.toolbox,
            mu=self.population_size,
            lambda_=self.population_size,
            cxpb=self.crossover_probability,
            mutpb=min(1.0 / self.num_variables, 0.2),
            ngen=self.num_generations,
            stats=self.stats,
            verbose=False,
        )

    # def create_individual_class(self):
    #     creator.create(
    #         "FitnessMin",
    #         base.Fitness,
    #         weights=(1.0,) * self.num_objectives,
    #         crowding_dist=0.0,
    #     )
    #     creator.create(
    #         "Individual", array.array, typecode="d", fitness=creator.FitnessMin
    #     )
    #
    # def create_model(
    #         self,
    #         problem,
    #         num_variables,
    #         population_size,
    #         lower_bound,
    #         upper_bound,
    #         crossover_probability,
    #         eta_crossover,
    #         eta_mutation,
    # ):
    #     self.create_individual_class()
    #
    #     toolbox = base.Toolbox()
    #     toolbox.register("attr_float", uniform, lower_bound, upper_bound, num_variables)
    #     toolbox.register(
    #         "individual", tools.initIterate, creator.Individual, toolbox.attr_float
    #     )
    #     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #     toolbox.register("evaluate", problem)
    #     toolbox.register(
    #         "mate",
    #         tools.cxSimulatedBinaryBounded,
    #         low=lower_bound,
    #         up=upper_bound,
    #         eta=eta_crossover,
    #     )
    #     toolbox.register(
    #         "mutate",
    #         tools.mutPolynomialBounded,
    #         low=lower_bound,
    #         up=upper_bound,
    #         eta=eta_mutation,
    #         indpb=min(1.0 / num_variables, 0.2),
    #     )
    #     toolbox.register("select", self.select)
    #
    #     self.toolbox = toolbox
    #
    #     self.stats = tools.Statistics(lambda ind: ind.fitness.values)
    #     self.stats.register("pop", copy.deepcopy)
    #
    # def select(self, individuals, k):
    #     chosen = tools.selNSGA2(individuals, k, nd=self.nd)
    #     self.print_stats(chosen)
    #     return chosen
    #
    # # def run(self):
    # #     pop = self.toolbox.population(n=self.population_size)
    # #     self.result_pop, self.logbook = algorithms.eaMuPlusLambda(
    # #         pop,
    # #         self.toolbox,
    # #         mu=self.population_size,
    # #         lambda_=self.population_size,
    # #         cxpb=self.crossover_probability,
    # #         mutpb=min(1.0 / self.num_variables, 0.2),
    # #         ngen=self.num_generations,
    # #         stats=self.stats,
    # #         verbose=False,
    # #     )
    #
    def print_stats(self, chosen=None):
        print("\n" + "=" * 80)
        print(f"Generation {self.current_generation}, population size: {len(chosen)}")
        self.current_generation += 1

        if not self.verbose:
            print("\n" + "=" * 80 + "\n")
            return

        # print(f"Population: {chosen}", end="\t")

        # if "hv" in self.log:
        #     print(f"HyperVolume: {self.hyper_volume(chosen)}", end="\t")

        # logbook = self.stats.compile(chosen)

        # for key in self.log:
        #     if key in logbook:
        #         print(f"{key}: {logbook[key]}", end="\t")

        print("\n" + "=" * 80 + "\n")

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
    #
    @staticmethod
    def varOr(population, toolbox, lambda_, cxpb, mutpb):
        offspring = []
        for _ in range(lambda_):
            op_choice = random.random()
            if op_choice < cxpb:  # Apply crossover
                ind1, ind2 = [toolbox.clone(i) for i in random.sample(population, 2)]
                ind1, ind2 = toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                offspring.append(ind1)
            elif op_choice < cxpb + mutpb:  # Apply mutation
                ind = toolbox.clone(random.choice(population))
                (ind,) = toolbox.mutate(ind)
                del ind.fitness.values
                offspring.append(ind)
            else:  # Apply reproduction
                offspring.append(random.choice(population))

        return offspring
    #
    def evaluate(self, population):
        for ind in population:
            ind.fitness.values = self.problem(ind)
    #
    # def run(self):
    #     pop = self.toolbox.population(n=self.population_size)
    #     self.result_pop, self.logbook = algorithms.eaMuPlusLambda(
    #         pop,
    #         self.toolbox,
    #         mu=self.population_size,
    #         lambda_=self.population_size,
    #         cxpb=self.crossover_probability,
    #         mutpb=min(1.0 / self.num_variables, 0.2),
    #         ngen=self.num_generations,
    #         stats=self.stats,
    #         verbose=False,
    #     )

    # def run(self):
    #     population = self.toolbox.population(n=self.population_size)
    #     self.evaluate(population)
    #
    #     logbook = tools.Logbook()
    #     logbook.header = ["gen", "nevals"] + self.stats.fields
    #
    #     record = self.stats.compile(population)
    #     logbook.record(gen=0, nevals=len(population), **record)
    #
    #     for g in range(1, self.num_generations + 1):
    #         self.evaluate(population)
    #         offspring = NSGA2.varOr(
    #             population,
    #             self.toolbox,
    #             self.population_size,
    #             self.crossover_probability,
    #             1.0 / self.num_variables,
    #         )
    #
    #         # combine offspring and population
    #         population = population + offspring
    #         self.evaluate(population)
    #         chosen = self.select(population, self.population_size)
    #         population = chosen
    #
    #         record = self.stats.compile(population)
    #         logbook.record(gen=g, nevals=len(population), **record)
    #
    #         self.print_stats(chosen)
    #
    #     self.result_pop = population
    #     self.logbook = logbook
