"""
@Author Sebastian Cucerca
@Created 08/03/2023
"""

from ._params2_bin import _params2_bin
from ._crossover_single import _crossover_single
from ._offset_rejected import _offset_rejected
from ._mutate import _mutate
from ._bin2_params import _bin2_params
from ._clip import _clip
import numpy as np
import numpy.typing as npt
from typing import List
import logging

# Create logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_random_parameters(
        limits: list,
        prec_facs: npt.NDArray,
        cont_ds: npt.NDArray,
        samp_cnt: int,
        min_ppl: int = None,
        max_t: float = None,
        hyp_lt: float = None,
) -> npt.NDArray:
    """
    TODO fill
    :param limits:
    :param prec_facs:
    :param cont_ds:
    :param samp_cnt:
    :param min_ppl:
    :param max_t:
    :param hyp_lt:
    :return:
    """

    #
    param_cnt = len(limits)

    #
    samp_parameters = np.zeros((0, param_cnt))

    #
    while samp_parameters.shape[0] < samp_cnt:

        #
        pot_samp_parameters = np.zeros((1, param_cnt))

        #
        for i in range(param_cnt):

            #
            curr_limits = limits[i]

            #
            if cont_ds[i]:

                #
                r = np.random.rand()

                #
                val = curr_limits[0] + r * (curr_limits[1] - curr_limits[0])

                #
                pot_samp_parameters[0, i] = np.round(val, len(str(prec_facs[i])) - 1)

            #
            else:

                #
                rand_idx = np.random.randint(len(curr_limits))

                #
                pot_samp_parameters[0, i] = curr_limits[rand_idx]

        #
        if min_ppl is not None:
            if _offset_rejected(pot_samp_parameters, min_ppl, max_t, hyp_lt):
                continue

        #
        samp_parameters = np.vstack((samp_parameters, pot_samp_parameters))

    #
    logger.info("Generated random starting population")

    #
    return samp_parameters


def generate_offsprings(
        pot_par_ds: npt.NDArray,
        new_pop_sz: int,
        cross_prob: float,
        mut_prob: float,
        scores: npt.NDArray,
        bits_per_param: npt.NDArray,
        prec_facs: npt.NDArray,
        limits: List[npt.NDArray],
        min_ppl: int = None,
        max_t: float = None,
        hyp_lt: float = None,
        divs: npt.NDArray = np.array([1000, 1, 1, 1, 1, 1, 1]),
) -> npt.NDArray:
    """
    TODO fill
    :param pot_par_ds:
    :param new_pop_sz:
    :param cross_prob:
    :param mut_prob:
    :param scores:
    :param bits_per_param:
    :param prec_facs:
    :param limits:
    :param min_ppl:
    :param max_t:
    :param hyp_lt:
    :param divs:
    :return:
    """

    #
    pot_par_ds_div = pot_par_ds.copy()

    #
    for i in range(len(divs)):
        pot_par_ds_div[:, i] /= divs[i]

    #
    pot_par_chroms = _params2_bin(pot_par_ds_div, bits_per_param, prec_facs)

    #
    off_ds = np.zeros((0, pot_par_ds.shape[1]), dtype=np.float32)

    #
    while off_ds.shape[0] < new_pop_sz:

        #
        pot_par_ids = np.arange(pot_par_ds.shape[0])
        par_ids = np.random.choice(pot_par_ids, 2, replace=False, p=scores)

        #
        par_chroms = pot_par_chroms[par_ids, :]

        #
        new_chrom = _crossover_single(par_chroms, cross_prob)

        #
        mut_chrom = _mutate(new_chrom)
        off_chrom = mut_chrom

        #
        pot_off_ds = _bin2_params(off_chrom, bits_per_param, prec_facs)

        #
        for i in range(len(divs)):
            pot_off_ds[:, i] *= divs[i]

        #
        pot_off_ds = _clip(pot_off_ds, limits)

        #
        already_offspring = False
        for i in range(off_ds.shape[0]):
            if np.array_equal(np.squeeze(pot_off_ds), off_ds[i, :]):
                already_offspring = True

        #
        already_parent = False
        for i in range(pot_par_ds.shape[0]):
            if np.array_equal(np.squeeze(pot_off_ds), pot_par_ds[i, :]):
                already_parent = True

        if already_offspring or already_parent:
            continue

        #
        if min_ppl is not None:
            if _offset_rejected(pot_off_ds, min_ppl, max_t, hyp_lt):
                continue

        #
        off_ds = np.concatenate((off_ds, pot_off_ds))

    #
    logger.info("Generated offspring generation [Size=%i]" % new_pop_sz)

    #
    return off_ds
