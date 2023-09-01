
"""
TODO Write file description

@Author Sebastian Cucerca
@Created 20/01/2023
"""

# Imports
import numpy as np
import numpy.typing as npt


def _spectrum_2_xyz(spectra: npt.NDArray,
					illuminant: npt.NDArray,
					xbar: npt.NDArray,
					ybar: npt.NDArray,
					zbar: npt.NDArray) -> npt.NDArray:

	# TODO explain
	norm_fac = 100 / np.sum(illuminant * ybar)

	# TODO explain
	x = norm_fac * spectra @ np.transpose(illuminant * xbar)
	y = norm_fac * spectra @ np.transpose(illuminant * ybar)
	z = norm_fac * spectra @ np.transpose(illuminant * zbar)

	#
	xyz_values = np.column_stack((x, y, z))

	#
	return xyz_values

"""
function XYZ = Spectra2XYZ(spectra, illuminant, xbar, ybar, zbar)
%  The input spectrum 'spectra' can be a vector or an n x 31 matrix.
%  With an n x 31 input matrix, an n x 3 output matrix is produced with one XYZ
%  value for each row of the input matrix
% xbar, ybar and zbar are the color matching functions (see xyzBarIlluminant.m)
% illuminant can be any illuminant with given spectral power distribution 
% or SPD (two illuminant SPDs given in xyzBarIlluminant.m)

%%%%%%%%%%%%%%%%%%%%%%%%%%%% BEGIN ASSIGNMENT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

norm_fac = 100 / sum(illuminant .* ybar);
X = norm_fac .* spectra * (illuminant .* xbar)';
Y = norm_fac .* spectra * (illuminant .* ybar)';
Z = norm_fac .* spectra * (illuminant .* zbar)';
XYZ = [X Y Z];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END ASSIGNMENT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end % function 
"""
