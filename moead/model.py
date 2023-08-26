from copy import deepcopy

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.moead import NeighborhoodSelection, default_decomp
from pymoo.core.algorithm import LoopwiseAlgorithm
from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.reference_direction import default_ref_dirs
from scipy.spatial.distance import cdist
from emoa.utils import *

from deap.tools._hypervolume import hv


class MOEAD(LoopwiseAlgorithm, GeneticAlgorithm):
    def __init__(
            self,
            pop_size=100,
            ref_dirs=None,
            n_neighbors=20,
            decomposition=None,
            prob_neighbor_mating=0.9,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=20),
            mutation=PM(prob_var=None, eta=20),
            output=MultiObjectiveOutput(),
            hv_ref=None,
            **kwargs
    ):
        self.ref_dirs = ref_dirs
        self.decomposition = decomposition
        self.n_neighbors = n_neighbors
        self.hv_ref = hv_ref
        self.neighbors = None

        self.hv_history = []

        self.selection = NeighborhoodSelection(prob=prob_neighbor_mating)

        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=NoDuplicateElimination(),
            output=output,
            advance_after_initial_infill=False,
            **kwargs
        )

    def _setup(self, problem, **kwargs):
        assert (
            not problem.has_constraints()
        ), "MOEAD does not support constrained problems"

        if self.ref_dirs is None:
            self.ref_dirs = default_ref_dirs(problem.n_obj)
        self.pop_size = len(self.ref_dirs)

        self.neighbors = np.argsort(
            cdist(self.ref_dirs, self.ref_dirs), axis=1, kind="quicksort"
        )[:, : self.n_neighbors]

        if self.decomposition is None:
            self.decomposition = default_decomp(problem)

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills, **kwargs)
        self.ideal = np.min(self.pop.get("F"), axis=0)

    def _next(self):
        pop = self.pop

        # iterate over all individuals in the population
        for k in np.random.permutation(len(pop)):
            # get the parents using the neighborhood selection
            P = self.selection.do(
                self.problem,
                pop,
                1,
                self.mating.crossover.n_parents,
                neighbors=[self.neighbors[k] - 1],
            )

            # perform a mating using the default operators - if more than one offspring just pick the first
            off = np.random.choice(
                self.mating.do(self.problem, pop, 1, parents=P, n_max_iterations=1)
            )

            # evaluate the offspring
            off = yield off

            # update the ideal point
            self.ideal = np.min(np.vstack([self.ideal, off.F]), axis=0)

            # now actually replace the individuals
            self._replace(k, off)

    def _replace(self, k, off):
        pop = self.pop

        # calculate the decomposed value for each neighbor
        N = self.neighbors[k] - 1
        FV = self.decomposition.do(
            pop[N].get("F"), weights=self.ref_dirs[N, :], ideal_point=self.ideal
        )
        off_FV = self.decomposition.do(
            off.F[None, :], weights=self.ref_dirs[N, :], ideal_point=self.ideal
        )

        I = np.where(off_FV < FV)[0]
        pop[N[I]] = off

        # print("hv: ", self.calc_hypervolume(pop, self.hv_ref))

        # self.hv_history.append(self.calc_hypervolume(pop, self.hv_ref))

    @staticmethod
    def calc_hypervolume(pop, ref=None):
        F = pop.get("F").astype(float, copy=False)
        fronts = NonDominatedSorting().do(F)
        objs = F[fronts[0]]
        if ref is None:
            ref = np.max(objs, axis=0) + 1
        return hv.hypervolume(objs, ref)

    @staticmethod
    def history_hypervolume(history, ref=None):
        return [MOEAD.calc_hypervolume(instance.pop, ref) for instance in history]
