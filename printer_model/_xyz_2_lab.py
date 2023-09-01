
"""
TODO Write file description

@Author Sebastian Cucerca
@Created 20/01/2023
"""

# Imports
import numpy as np
import numpy.typing as npt


def _xyz_2_lab(xyz_values: npt.NDArray, white_reference: npt.NDArray) -> npt.NDArray:
    """
    Converts input values from CIEXYZ space to CIEL*a*b* space
    :param xyz_values: Input values in CIEXYZ space
    :param white_reference: White refence point in CIEXYZ space
    :return: Output values in CIEL*a*b* space
    """
    if len(xyz_values.shape)==1:
        xyz_values = xyz_values[:, None]
    if len(white_reference.shape)==1:
        white_reference = white_reference[None, :]
    
    eps=0.008856
    xxn, yyn, zzn = np.split(xyz_values / white_reference, (1, 2), 1)
    fxxn = np.where(xxn > eps, xxn ** (1 / 3), 7.787 * xxn + 16 / 116)
    fyyn = np.where(yyn > eps, yyn ** (1 / 3), 7.787 * yyn + 16 / 116)
    L = np.where(yyn > eps, 116 * fyyn - 16, 903.3 * yyn)
    fzzn = np.where(zzn > eps, zzn ** (1 / 3), 7.787 * zzn + 16 / 116)
    a = 500 * (fxxn - fyyn)
    b = 200 * (fyyn - fzzn)

    Lab = np.concatenate([L, a, b], 1)
    return Lab
    
"""
function Lab = XYZ2Lab(XYZ, refWhite)

    eps = 0.008856;

    xyz_norm = XYZ ./ repmat(refWhite, size(XYZ, 1), 1);


    f1 = xyz_norm.^(1/3);
    f2 = 7.787 .* xyz_norm + (16/116);

    f3 = f2;
    xyz_mask = (xyz_norm > eps);
    f3(xyz_mask) = f1(xyz_mask);

    L = 116 .* f1(:, 2) - 16;
    a = 500 .* (f3(:, 1) - f3(:, 2));
    b = 200 .* (f3(:, 2) - f3(:, 3));

    Lab = [L a b];

end
"""
