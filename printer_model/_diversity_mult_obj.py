
"""
@Author Sebastian Cucerca
@Created 20/01/2023
"""

# Imports
import numpy as np
import numpy.typing as npt
import scipy.spatial
from typing import List


def _diversity_mult_obj(parameter_values: npt.NDArray, k: int, limits: List[npt.NDArray]) -> npt.NDArray:
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
	distances_linear = scipy.spatial.distance.pdist(normalized_parameter_values)
	distances_square = scipy.spatial.distance.squareform(distances_linear)

	# Set distance between each point to itself to infinity
	# since we want to extract only distances to other points
	np.fill_diagonal(distances_square, np.inf)

	# Sort distance matrix
	distances_square_sorted = np.sort(distances_square, axis=1)

	# Get k the closest points for each point and compute their average distance
	diversity_scores = np.mean(distances_square_sorted[:, :k], axis=1)

	# Return performance scores
	return diversity_scores


"""
function vals = DiversityMultObj(pts, k, limitsDS)
    mxVals = cell2mat(cellfun(@max,limitsDS, 'Uni', 0));
    normPts = pts ./ repmat(mxVals, size(pts, 1), 1);
    dists = pdist(normPts);
    dists = squareform(dists);
    dists(eye(size(dists))==1) = Inf; 
    vals = mean(mink(dists, k, 2), 2);
end


"""
