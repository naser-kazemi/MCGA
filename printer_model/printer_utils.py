from scipy.spatial import ConvexHull
from emoa.utils import *
from skimage import color
from deap import base, creator, tools, algorithms


def compute_area(xyz_colors) -> float:
    xy_colors = xyz_colors[:, :2]

    area = ConvexHull(xy_colors).volume

    return area


def compute_neugebauer(colorant_area_coverage: npt.NDArray, ng_primary_reflectances: npt.NDArray) -> npt.NDArray:
    predicted_spectrum = np.matmul(colorant_area_coverage, ng_primary_reflectances)
    return predicted_spectrum


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


def xyz_2_lab(xyz_values: npt.NDArray, white_reference: npt.NDArray) -> npt.NDArray:
    """
    Converts input values from CIEXYZ space to CIEL*a*b* space
    :param xyz_values: Input values in CIEXYZ space
    :param white_reference: White refence point in CIEXYZ space
    :return: Output values in CIEL*a*b* space
    """
    if len(xyz_values.shape) == 1:
        xyz_values = xyz_values[:, None]
    if len(white_reference.shape) == 1:
        white_reference = white_reference[None, :]

    eps = 0.008856
    xxn, yyn, zzn = np.split(xyz_values / white_reference, (1, 2), 1)
    fxxn = np.where(xxn > eps, xxn ** (1 / 3), 7.787 * xxn + 16 / 116)
    fyyn = np.where(yyn > eps, yyn ** (1 / 3), 7.787 * yyn + 16 / 116)
    L = np.where(yyn > eps, 116 * fyyn - 16, 903.3 * yyn)
    fzzn = np.where(zzn > eps, zzn ** (1 / 3), 7.787 * zzn + 16 / 116)
    a = 500 * (fxxn - fyyn)
    b = 200 * (fyyn - fzzn)

    Lab = np.concatenate([L, a, b], 1)
    return Lab


def spectrum_2_xyz(spectra: npt.NDArray,
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


def predict_printer_colors(
        ink_area_coverages: npt.NDArray,
        ng_primary_reflectances: npt.NDArray,
        white_reference: npt.NDArray,
        d65_illuminant: npt.NDArray,
        xbar: npt.NDArray,
        ybar: npt.NDArray,
        zbar: npt.NDArray,
) -> tuple:
    colorant_area_coverages = compute_demichel(ink_area_coverages)
    predicted_spectrum = compute_neugebauer(colorant_area_coverages, ng_primary_reflectances)

    # Convert spectra to XYZ
    predicted_xyz_colors = spectrum_2_xyz(
        predicted_spectrum, d65_illuminant, xbar, ybar, zbar
    )

    # Convert XYZ to Lab
    predicted_lab_colors = xyz_2_lab(predicted_xyz_colors, white_reference)

    predicted_rgb_colors = np.zeros(predicted_lab_colors.shape, dtype=np.float32)
    for i in range(predicted_lab_colors.shape[0]):
        predicted_rgb_colors[i, :] = color.lab2rgb(
            [
                predicted_lab_colors[i, 0],
                predicted_lab_colors[i, 1],
                predicted_lab_colors[i, 2],
            ]
        )

    return predicted_rgb_colors, predicted_xyz_colors, predicted_lab_colors


def plot_lab_gamut(lab_colors: npt.NDArray):
    #
    for i in range(lab_colors.shape[0]):
        #
        lab_color = lab_colors[i, :]

        #
        rgb_color = color.lab2rgb([lab_color[0], lab_color[1], lab_color[2]])

        #
        plt.scatter(lab_color[1],
                    lab_color[2],
                    color=np.array(rgb_color))

    #
    plt.xlabel('CIE-a*')
    plt.ylabel('CIE-b*')

    #
    plt.xlim(-60, 60)
    plt.ylim(-60, 60)

    #
    plt.show()


def save_lab_gamut(lab_colors: npt.NDArray, file_path: str, file_name: str = "plot", title: str = ""):
    #
    plt.figure()

    #
    colors = np.zeros((len(lab_colors), 3))

    #
    for i in range(lab_colors.shape[0]):
        #
        lab_color = lab_colors[i, :]

        #
        colors[i, :] = color.lab2rgb([lab_color[0], lab_color[1], lab_color[2]])

    #
    a_values = [lab_colors[i, 1] for i in range(lab_colors.shape[0])]
    b_values = [lab_colors[i, 2] for i in range(lab_colors.shape[0])]

    #
    plt.scatter(a_values, b_values, color=colors, s=20)

    #
    plt.xlabel('CIE-a*')
    plt.ylabel('CIE-b*')

    #
    plt.title(title)

    #
    plt.xlim(-60, 60)
    plt.ylim(-60, 60)

    #
    plt.savefig('%s/%s.png' % (file_path, file_name), dpi=300)

    #
    plt.clf()

    #
    plt.close()


def get_random_parameters(
        limits: list,
        prec_facs: npt.NDArray,
        cont_ds: npt.NDArray,
        samp_cnt: int,
        min_ppl: int = None,
        max_t: float = None,
        hyp_lt: float = None,
) -> npt.NDArray:
    """
    TODO fill
    :param limits:
    :param prec_facs:
    :param cont_ds:
    :param samp_cnt:
    :param min_ppl:
    :param max_t:
    :param hyp_lt:
    :return:
    """

    #
    param_cnt = len(limits)

    #
    samp_parameters = np.zeros((0, param_cnt))

    #
    while samp_parameters.shape[0] < samp_cnt:

        #
        pot_samp_parameters = np.zeros((1, param_cnt))

        #
        for i in range(param_cnt):

            #
            curr_limits = limits[i]

            #
            if cont_ds[i]:

                #
                r = np.random.rand()

                #
                val = curr_limits[0] + r * (curr_limits[1] - curr_limits[0])

                #
                pot_samp_parameters[0, i] = np.round(val, len(str(prec_facs[i])) - 1)

            #
            else:

                #
                rand_idx = np.random.randint(len(curr_limits))

                #
                pot_samp_parameters[0, i] = curr_limits[rand_idx]

        #
        # if min_ppl is not None:
        #     if _offset_rejected(pot_samp_parameters, min_ppl, max_t, hyp_lt):
        #         continue

        #
        samp_parameters = np.vstack((samp_parameters, pot_samp_parameters))

    #
    return samp_parameters


def varOr(population, toolbox, lambda_, cxpb, mutpb):
    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:  # Apply crossover
            ind1, ind2 = [toolbox.clone(i) for i in random.sample(population, 2)]
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:  # Apply reproduction
            offspring.append(random.choice(population))

    return offspring
