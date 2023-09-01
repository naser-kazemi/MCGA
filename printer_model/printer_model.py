import exploration_params
from oraclase.color_processing._xyz_2_lab import _xyz_2_lab
import oraclase.color_processing as color_processing
import oraclase.visualization as visualization
import oraclase.genetic_algorithm as genetic_algorithm
import oraclase.sorting as sorting
from _diversity_mult_obj import _diversity_mult_obj
from ._chromaticity_obj import _chromaticity_obj
import numpy as np
import scipy
from skimage import color
import numpy.typing as npt
import os
import matplotlib.pyplot as plt


def predict_printer_colors(
    ink_area_coverages: npt.NDArray,
    ng_primary_reflectances: npt.NDArray,
    white_reference: npt.NDArray,
    d65_illuminant: npt.NDArray,
    xbar: npt.NDArray,
    ybar: npt.NDArray,
    zbar: npt.NDArray,
):
    #
    colorant_area_coverages = compute_demichel(ink_area_coverages)

    # Use Neugebauer model to predict spectra
    predicted_spectras = compute_neugebauer(
        colorant_area_coverages, ng_primary_reflectances
    )

    # Convert spectra to XYZ
    predicted_xyz_colors = color_processing._spectrum_2_xyz(
        predicted_spectras, d65_illuminant, xbar, ybar, zbar
    )

    # Convert XYZ to Lab
    predicted_lab_colors = _xyz_2_lab(predicted_xyz_colors, white_reference)

    #
    predicted_rgb_colors = np.zeros(predicted_lab_colors.shape, dtype=np.float32)
    for i in range(predicted_lab_colors.shape[0]):
        predicted_rgb_colors[i, :] = color.lab2rgb(
            [
                predicted_lab_colors[i, 0],
                predicted_lab_colors[i, 1],
                predicted_lab_colors[i, 2],
            ]
        )

    return [predicted_rgb_colors, predicted_xyz_colors, predicted_lab_colors]


def compute_demichel(ink_area_coverage: npt.NDArray) -> npt.NDArray:
    c = ink_area_coverage[:, 0]
    m = ink_area_coverage[:, 1]
    y = ink_area_coverage[:, 2]

    aw = (1 - c) * (1 - m) * (1 - y)
    ac = c * (1 - m) * (1 - y)
    am = (1 - c) * m * (1 - y)
    ay = (1 - c) * (1 - m) * y
    ar = (1 - c) * m * y
    ag = c * (1 - m) * y
    ab = c * m * (1 - y)
    ak = c * m * y

    colorant_area_coverage = np.stack((aw, ac, am, ay, ar, ag, ab, ak), axis=-1)

    return colorant_area_coverage


def compute_neugebauer(
    colorant_area_coverage: npt.NDArray, ng_primary_reflectances: npt.NDArray
) -> npt.NDArray:
    predicted_spectras = np.matmul(colorant_area_coverage, ng_primary_reflectances)

    return predicted_spectras


def compute_area(xyz_colors) -> float:
    xy_colors = xyz_colors[:, :2]

    area = scipy.spatial.ConvexHull(xy_colors).volume

    return area


if __name__ == "__main__":

    """
    To run the printer model you need to comment out the
    division/multiplication in line 125 and 152 of the genetic algorithm script.
    """

    #
    plot_dir = os.path.join("plots", exploration_params.name)
    os.makedirs(plot_dir, exist_ok=True)

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

    visualization.save_lab_gamut(
        test_lab_colors,
        os.path.join("plots", exploration_params.name),
        "gamut_full",
        "Full printer gamut",
    )

    ### Run full exploration

    # Create list with all samples
    all_samples = []

    # Create exploration

    # Create initial collection
    points_ds_p0 = genetic_algorithm.get_random_parameters(
        exploration_params.limits_ds,
        exploration_params.prec_facs,
        exploration_params.cont_ds,
        exploration_params.pop_size,
    )

    points_ds_p0 *= 0.001

    # Compute predicted colors for initial collection
    [_, xyz_colors_p0, points_ps_p0] = predict_printer_colors(
        points_ds_p0,
        exploration_params.ng_primary_reflectances,
        exploration_params.white_ciexyz,
        exploration_params.d65_illuminant,
        exploration_params.xbar,
        exploration_params.ybar,
        exploration_params.zbar,
    )

    # Compute and visualize gamut area of test samples
    p0_area = compute_area(xyz_colors_p0)

    #
    areas = np.zeros(exploration_params.iterations + 1)
    areas[0] = p0_area

    #
    print("Gamut area of initial collection is %.6f" % p0_area)

    #
    visualization.save_lab_gamut(
        points_ps_p0, plot_dir, "gamut_a_initial", "Initial gamut (area=%.3f)" % p0_area
    )

    #
    points_ps_all = points_ps_p0.copy()
    all_xyz_colors = xyz_colors_p0.copy()

    #
    points_ds_pi = points_ds_p0
    points_ps_pi = points_ps_p0

    # Run exploration
    for i in range(exploration_params.iterations):
        #
        obj1 = _chromaticity_obj(points_ps_pi[:, 1:3])
        obj2 = _diversity_mult_obj(
            points_ds_pi, exploration_params.k_n, exploration_params.limits_ds
        )
        obj3 = _diversity_mult_obj(
            points_ps_pi[:, 1:3],
            exploration_params.k_n,
            exploration_params.limits_ps[1:3],
        )
        obj_scores_pi = np.column_stack((obj1, obj2, obj3))

        #
        _, scores_pi, _, _ = sorting.sort_nd(
            points_ps_pi,
            exploration_params.start_angle_range,
            exploration_params.cone_count_range,
            obj_scores_pi,
        )

        #
        scores_pi = scores_pi / np.sum(scores_pi)

        #
        points_ds_qi = genetic_algorithm.generate_offsprings(
            points_ds_pi,
            exploration_params.pop_size,
            exploration_params.cross_prob,
            exploration_params.mut_prob,
            scores_pi,
            exploration_params.bits_per_param,
            exploration_params.prec_facs,
            exploration_params.limits_ds,
            divs=np.array([1, 1, 1]),
        )

        # Compute predicted colors for offspring collection
        [_, xyz_colors_qi, points_ps_qi] = predict_printer_colors(
            points_ds_qi,
            exploration_params.ng_primary_reflectances,
            exploration_params.white_ciexyz,
            exploration_params.d65_illuminant,
            exploration_params.xbar,
            exploration_params.ybar,
            exploration_params.zbar,
        )

        #
        points_ps_all = np.vstack((points_ps_all, points_ps_qi))
        all_xyz_colors = np.vstack((all_xyz_colors, xyz_colors_qi))

        # Compute and visualize gamut area of test samples
        current_area = compute_area(all_xyz_colors)

        areas[i + 1] = current_area

        print(
            "Gamut area of all samples (after iteration=%i) is %.6f" % (i, current_area)
        )

        visualization.save_lab_gamut(
            points_ps_all,
            plot_dir,
            "gamut_after_i%i" % i,
            "Gamut (after iteration=%i, area=%.3f)" % (i, current_area),
        )

        #
        points_ds_ri = np.vstack((points_ds_pi, points_ds_qi))
        points_ps_ri = np.vstack((points_ps_pi, points_ps_qi))

        #
        obj1 = _chromaticity_obj(points_ps_ri[:, 1:3])
        obj2 = _diversity_mult_obj(
            points_ds_ri, exploration_params.k_n, exploration_params.limits_ds
        )
        obj3 = _diversity_mult_obj(
            points_ps_ri[:, 1:3],
            exploration_params.k_n,
            exploration_params.limits_ps[1:3],
        )
        obj_scores_ri = np.column_stack((obj1, obj2, obj3))

        #
        _, _, sorted_ids_ri, _ = sorting.sort_nd(
            points_ps_ri,
            exploration_params.start_angle_range,
            exploration_params.cone_count_range,
            obj_scores_ri,
        )

        #
        ids_pni = sorted_ids_ri[: exploration_params.pop_size]

        #
        points_ds_pni = points_ds_ri[ids_pni, :]
        points_ps_pni = points_ps_ri[ids_pni, :]

        #
        points_ds_pi = points_ds_pni
        points_ps_pi = points_ps_pni

    #
    plt.figure()
    plt.plot(np.arange(exploration_params.iterations + 1), areas, label="Exploration")
    plt.plot(
        np.arange(exploration_params.iterations + 1),
        np.repeat(test_area, exploration_params.iterations + 1),
        label="Base Line",
    )
    plt.title("Gamut Areas")
    plt.xlabel("Iteration")
    plt.ylabel("Area")
    plt.legend(loc="lower right")
    plt.savefig("%s/%s.png" % (plot_dir, "area_growth"))
