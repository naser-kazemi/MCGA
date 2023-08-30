"""
@Author Sebastian Cucerca
@Created 20/01/2023
"""

from ._pol_to_cart import _pol_to_cart
from ._construct_fronts import _construct_fronts
from ._lt_obj import _lt_obj
from ._l_star_obj import _l_star_obj
from ._diversity_mult_obj import _diversity_mult_obj
from ._chromaticity_obj import _chromaticity_obj
import numpy as np
from scipy.spatial import Delaunay
import numpy.typing as npt
from typing import List
import logging

# Create logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sort_nd(points_ps, start_angle_range, cone_count_range, obj_scores):
    """
    TODO fill
    TODO add types
    """

    # compute fronts using hue wheel
    mc_samps = 0
    cone_len = 100

    num_points_ps = points_ps.shape[0]
    point_fronts = np.zeros((num_points_ps, num_points_ps))
    norm_point_fronts = np.zeros((num_points_ps, num_points_ps))

    avg_diff_thresh = 0.001
    min_mc_samps = 20
    max_mc_samps = 1000
    avg_diff = np.inf
    pi = np.pi

    while (avg_diff > avg_diff_thresh) or (mc_samps < min_mc_samps):

        if mc_samps >= max_mc_samps:
            break

        # random sample cone start position and size
        # compute number of bins
        cr = np.random.rand()
        ora = np.random.rand()

        #
        start_angle = start_angle_range[0] + ora * (
                start_angle_range[1] - start_angle_range[0]
        )
        cone_count = cone_count_range[0] + round(
            cr * (cone_count_range[1] - cone_count_range[0])
        )
        rad_per_cone = 2 * pi / cone_count

        # iterate over cones and construct fronts
        for c_idx in range(cone_count):

            # extract points within current cone
            clx, cly = _pol_to_cart(start_angle + (c_idx) * rad_per_cone, cone_len)
            crx, cry = _pol_to_cart(start_angle + (c_idx + 1) * rad_per_cone, cone_len)
            poly = np.array([[0, 0], [clx, cly], [crx, cry], [0, 0]])

            point = np.column_stack([points_ps[:, 1], points_ps[:, 2]])
            # https://stackoverflow.com/a/60672266
            in_mask = Delaunay(poly).find_simplex(point)
            cone_points_i_ds = np.where(in_mask != -1)[0]

            if not len(cone_points_i_ds):
                continue

            # compute scores for cone points
            curr_obj_scores = obj_scores[cone_points_i_ds, :]
            fronts = _construct_fronts(curr_obj_scores)

            # count occurances in front positions for each point
            for f_idx in range(fronts.shape[0]):
                loc_p_ids = np.nonzero(fronts[f_idx, :])
                glob_p_ids = cone_points_i_ds[loc_p_ids]
                point_fronts[glob_p_ids, f_idx] = point_fronts[glob_p_ids, f_idx] + 1

        new_norm_point_fronts = point_fronts / point_fronts.sum(1, keepdims=True)
        diffs = abs(new_norm_point_fronts - norm_point_fronts)
        avg_diff = diffs.mean()
        norm_point_fronts = new_norm_point_fronts

        mc_samps = mc_samps + 1

    # sort points according to
    print("Computed fronts (#%i hue wheel samplings, %.5f)" % (mc_samps, avg_diff))
    # we want to sort in descending order while lexsort sorts in ascending
    # we know that pointfronts is positive, so we can negate it to
    # get the desired behavior
    sort_ids = np.lexsort(-point_fronts.T[::-1])
    # compute ranks for points
    ranks = np.zeros(
        (
            len(
                points_ps,
            )
        ),
        dtype=np.int32,
    )
    ranks[sort_ids] = np.arange(len(points_ps))

    # # compute scores for points
    scores = (points_ps.shape[0] - ranks) / points_ps.shape[0]

    # # compute array with sorted ids according to ranks
    sorted_ids = ranks.argsort()

    # Normalize scores
    scores = scores / np.sum(scores)

    #
    logger.info("Sorted population")

    return ranks, scores, sorted_ids, point_fronts


def compute_score(
        points_ds: npt.NDArray,
        points_ps: npt.NDArray,
        obj: npt.NDArray,
        k_n: int,
        limits_ds: List[npt.NDArray],
        limits_ps: List[npt.NDArray],
):
    """
    TODO Fill
    :param points_ds:
    :param points_ps:
    :param obj:
    :param k_n:
    :param limits_ds:
    :param limits_ps:
    :return:
    """

    #
    assert len(obj) == 5

    #
    assert np.sum(obj) >= 1
    if len(points_ds.shape) == 1:
        points_ds = points_ds[None, :]
    if len(points_ps.shape) == 1:
        points_ps = points_ps[None, :]

    #
    obj1 = _chromaticity_obj(points_ps[:, 1:3])

    #
    obj2 = _diversity_mult_obj(points_ds, k_n, limits_ds)

    #
    obj3 = _diversity_mult_obj(points_ps[:, 1:3], k_n, limits_ps[1:3])

    #
    obj4 = _lt_obj(points_ps[:, 3])

    #
    obj5 = _l_star_obj(points_ps[:, 0])

    #
    scores = np.column_stack((obj1, obj2, obj3, obj4, obj5))

    #
    scores = scores * np.tile(obj, (points_ds.shape[0], 1))

    #
    scores = scores[:, obj != 0]

    #
    logger.info("Computed scores of population")

    #
    return scores
