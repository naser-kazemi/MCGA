from emoa.utils import *
from scipy.spatial import distance


def _lt_obj(lts: npt.NDArray) -> npt.NDArray:
    """
    TODO Write explanation
    :param lts:
    :return:
    """
    ids = np.argsort(lts)
    scores = np.zeros(len(lts))
    scores[ids] = np.linspace(1, len(lts), len(lts))
    scores = scores / len(lts)

    return scores


def _l_star_obj(l_star_values: npt.NDArray) -> npt.NDArray:
    """
    TODO Write explanation
    :param l_star_values:
    :return:
    """
    obj_values = l_star_values / 100
    return obj_values


def diversity_mult_obj(
        parameter_values: npt.NDArray, k: int, limits: list[npt.NDArray]
) -> npt.NDArray:
    """
    Computes average distance of each point to its closest k points
    :param parameter_values:
    :param k: Defines how many closest points are considered
    :param limits:
    :return: Average distances for each point
    """

    #
    limits_mxs = [l.max() for l in limits]

    #
    limits_expanded = np.tile(limits_mxs, (parameter_values.shape[0], 1))

    #
    normalized_parameter_values = parameter_values / limits_expanded

    # Compute distance matrix
    distances_linear = distance.pdist(normalized_parameter_values)
    distances_square = distance.squareform(distances_linear)

    # Set distance between each point to itself to infinity
    # since we want to extract only distances to other points
    np.fill_diagonal(distances_square, np.inf)

    # Sort distance matrix
    distances_square_sorted = np.sort(distances_square, axis=1)

    # Get k the closest points for each point and compute their average distance
    diversity_scores = np.mean(distances_square_sorted[:, :k], axis=1)

    # Return performance scores
    return diversity_scores


def chromaticity_obj(ab_star_values: npt.NDArray) -> npt.NDArray:
    """
    TODO Write explanation
    :param ab_star_values:
    :return:
    """

    # Compute chromaticity
    chromaticities = np.linalg.norm(ab_star_values, axis=1)

    #
    return chromaticities


def compute_score(
        points_ds: npt.NDArray,
        points_ps: npt.NDArray,
        obj: npt.NDArray,
        k_n: int,
        limits_ds: list[npt.NDArray],
        limits_ps: list[npt.NDArray],
):
    assert len(obj) == 5

    assert np.sum(obj) >= 1

    if len(points_ds.shape) == 1:
        points_ds = points_ds[None, :]
    if len(points_ps.shape) == 1:
        points_ps = points_ps[None, :]

    obj1 = chromaticity_obj(points_ps[:, 1:3])
    obj2 = diversity_mult_obj(points_ds, k_n, limits_ds)
    obj3 = diversity_mult_obj(points_ps[:, 1:3], k_n, limits_ps[1:3])
    obj4 = _lt_obj(points_ps[:, 3])
    obj5 = _l_star_obj(points_ps[:, 0])

    scores = np.column_stack((obj1, obj2, obj3, obj4, obj5))
    scores = scores * np.tile(obj, (points_ds.shape[0], 1))
    scores = scores[:, obj != 0]

    return scores
