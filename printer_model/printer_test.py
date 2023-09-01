from . import exploration_params
from .printer_objectives import *

from scipy.spatial import ConvexHull
from emoa.utils import *
from skimage import color
from .printer_utils import *


def main():
    plot_dir = os.path.join("printer_plots", exploration_params.name)
    os.makedirs(plot_dir, exist_ok=True)
    print(plot_dir)

    # Sample design space space
    smp = np.linspace(0, 1, 10)
    sets = np.stack((smp, smp, smp), axis=1)
    points_ds_test = np.array(np.meshgrid(smp, smp, smp)).T.reshape(-1, 3)

    # Compute predicted colors for samples
    [test_rgb_colors, test_xyz_colors, test_lab_colors] = predict_printer_colors(
        points_ds_test,
        exploration_params.ng_primary_reflectances,
        exploration_params.white_ciexyz,
        exploration_params.d65_illuminant,
        exploration_params.xbar,
        exploration_params.ybar,
        exploration_params.zbar,
    )

    # Compute and visualize gamut area of test samples
    test_area = compute_area(test_xyz_colors)
    print("Gamut area of test samples is %.6f" % test_area)

    save_lab_gamut(
        test_lab_colors,
        plot_dir,
        "gamut_full",
        "Full printer gamut",
    )

    points_ds_p0 = get_random_parameters(
        exploration_params.limits_ds,
        exploration_params.prec_facs,
        exploration_params.cont_ds,
        exploration_params.pop_size,
    ) * 0.001


if __name__ == "__main__":
    main()
