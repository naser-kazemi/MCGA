"""
@Author Sebastian Cucerca
@Created 20/01/2023
"""

# Imports
import numpy as np
import numpy.typing as npt


def _chromaticity_obj(ab_star_values: npt.NDArray) -> npt.NDArray:
    """
    TODO Write explanation
    :param ab_star_values:
    :return:
    """

    # Compute chromaticity
    chromaticities = np.linalg.norm(ab_star_values, axis=1)

    #
    return chromaticities


"""
function vals = ChromaticityObj(pts) 
    vals = vecnorm(pts, 2, 2);
    disp('Evaluated Chromaticity objective');
end


"""
