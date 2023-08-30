
"""
@Author Sebastian Cucerca
@Created 20/01/2023
"""

# Imports
import numpy as np
import numpy.typing as npt


def _l_star_obj(l_star_values: npt.NDArray) -> npt.NDArray:
	"""
	TODO Write explanation
	:param l_star_values:
	:return:
	"""

	#
	obj_values = l_star_values / 100

	#
	return obj_values


"""
function vals = LStarObj(Ls)
    vals = Ls ./ 100;
end


"""
