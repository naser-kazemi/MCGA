
"""
@Author Sebastian Cucerca
@Created 20/01/2023
"""

# Imports
import numpy as np
import numpy.typing as npt


def _lt_obj(lts: npt.NDArray) -> npt.NDArray:
    """
    TODO Write explanation
    :param lts:
    :return:
    """

    #
    ids = np.argsort(lts)

    #
    scores = np.zeros(len(lts))

    #
    scores[ids] = np.linspace(1, len(lts), len(lts))

    #
    scores = scores / len(lts)

    #
    return scores


"""
function scores = LTObj(lts)
    
    [~, ids] = sort(lts, 'ascend');
    
    scores = zeros(numel(lts), 1);
    scores(ids) = 1:numel(lts);
    scores = scores ./ numel(lts);
    
end


"""
