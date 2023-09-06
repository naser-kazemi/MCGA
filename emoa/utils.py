import io

import PIL
import numpy as np
import numpy.typing as npt
import argparse as ap
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import random

from matplotlib.figure import Figure


def create_parser():
    """
    Create the parser for the command line arguments to create a new NSGA2 instance
    """
    args = ap.ArgumentParser()
    args.add_argument(
        "--population-size", type=int, default=100, help="The size of the population"
    )
    args.add_argument(
        "--num-variables", type=int, default=2, help="The number of variables"
    )
    args.add_argument(
        "--num-objectives", type=int, default=2, help="The number of objectives"
    )
    args.add_argument(
        "--num-generations", type=int, default=100, help="The number of generations"
    )
    args.add_argument(
        "--tournament-size", type=int, default=2, help="The tournament size"
    )
    args.add_argument(
        "--eta-crossover", type=float, default=1.0, help="The eta crossover"
    )
    args.add_argument(
        "--eta-mutation", type=float, default=1.0, help="The eta mutation"
    )
    args.add_argument(
        "--crossover-probability",
        type=float,
        default=0.9,
        help="The crossover probability",
    )
    args.add_argument(
        "--output-dir", type=str, default="output", help="The output directory"
    )
    return args.parse_args()


def create_gif(images_directory="gif_images", output_file="output.gif"):
    """
    Create a GIF from the gif_images in the gif_images directory
    """

    # get the list of gif_images in the gif_images directory
    images_list = os.listdir(images_directory)
    # sort the list of gif_images
    images_list.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))

    frames = []
    # read all gif_images in the gif_images directory and add them to the frames list
    for image in images_list:
        frames.append(imageio.v2.imread(os.path.join(images_directory, image)))

    # create the GIF
    imageio.mimsave(output_file, frames, duration=200)


def cartesian_to_polar(x):
    """
    Convert the cartesian coordinates to polar coordinates
    :param x: the cartesian coordinates
    :return: the polar coordinates
    """

    x1 = x[:, 0]
    x2 = x[:, 1]

    r = np.sqrt(x1**2 + x2**2)
    theta = np.arctan2(x2, x1)
    # theta = np.arctan(x2 / x1)

    return r, theta


def to_polar(_x):
    """
    Convert the n-dimensional cartesian coordinates to polar coordinates
    :param _x: the n-dimensional cartesian coordinates
    :return: the n-dimensional polar coordinates
    """

    x = np.copy(_x)

    # compute the radius
    r = np.sqrt(np.sum(x**2, axis=1))

    # compute the angles
    theta = np.zeros((x.shape[0], x.shape[1] - 1))

    theta[:, 0] = np.arctan2(x[:, 1], x[:, 0])
    x[:, 1] = x[:, 1] / np.sin(theta[:, 0])
    for i in range(1, x.shape[1] - 1):
        theta[:, i] = np.arctan2(x[:, i], x[:, i + 1])
        x[:, i + 1] = x[:, i + 1] / np.cos(theta[:, i])

    return r, theta


def to_cartesian(_r, _theta):
    """
    Convert the n-dimensional polar coordinates to cartesian coordinates
    :param _r: the radius
    :param _theta: the n-dimensional angles
    :return: the n-dimensional cartesian coordinates
    """

    r = np.copy(_r)
    theta = np.copy(_theta)

    # compute the x coordinates
    x = np.zeros((theta.shape[0], theta.shape[1] + 1))
    sin_prod = np.prod(np.sin(theta[:, 1:]), axis=1)

    x[:, 0] = r * np.cos(theta[:, 0]) * sin_prod
    x[:, 1] = r * np.sin(theta[:, 0]) * sin_prod

    for i in range(2, theta.shape[1] + 1):
        sin_prod = sin_prod / np.sin(theta[:, i - 1])
        x[:, i] = r * np.cos(theta[:, i - 1]) * sin_prod

    return x


def vector_to_polar(_x):
    """
    Convert the n-dimensional cartesian coordinates to polar coordinates
    :param _x: the n-dimensional cartesian coordinates
    :return: the n-dimensional polar coordinates
    """

    x = np.copy(_x)

    # compute the radius
    r = np.sqrt(np.sum(x**2))

    # compute the angles
    theta = np.zeros((len(x) - 1))

    theta[0] = np.arctan2(x[1], x[0])
    x[1] = x[1] / np.sin(theta[0]) if np.sin(theta[0]) != 0 else 0
    for i in range(1, len(theta)):
        theta[i] = np.arctan2(x[i], x[i + 1])
        x[i + 1] = x[i + 1] / np.cos(theta[i]) if np.cos(theta[i]) != 0 else 0

    for i in range(len(theta)):
        if theta[i] < 0:
            theta[i] += 2 * np.pi

    return r, theta


def vector_to_cartesian(r, _theta):
    """
    Convert the n-dimensional polar coordinates to cartesian coordinates
    :param r: the radius
    :param _theta: the n-dimensional angles
    :return: the n-dimensional cartesian coordinates
    """

    theta = np.copy(_theta)

    # compute the x coordinates
    x = np.zeros((len(theta) + 1))
    sin_prod = np.prod(np.sin(theta[1:]))

    x[0] = r * np.cos(theta[0]) * sin_prod
    x[1] = r * np.sin(theta[0]) * sin_prod

    for i in range(2, len(x)):
        if np.sin(theta[i - 1]) != 0:
            sin_prod = sin_prod / np.sin(theta[i - 1])
        else:
            sin_prod = 1
        x[i] = r * np.cos(theta[i - 1]) * sin_prod

    return x


def generate_color():
    """
    Generate a random color in base16 format
    :return: the random color
    """

    color = "#"
    for i in range(6):
        color += random.choice("0123456789ABCDEF")
    return color


EPSILON = 1e-2


def has_duplicate_member(population):
    """
    Check if the population has a duplicate member
    :param population: the population
    :return: True if the population has a duplicate member, False otherwise
    """

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            if np.all(
                np.abs(
                    np.subtract(
                        population[i].objective_values, population[j].objective_values
                    )
                )
                < 1e-5
            ):
                return True
    return False


def has_duplicate_individuals(individuals):
    for i in range(len(individuals)):
        for j in range(i + 1, len(individuals)):
            if individuals[i].fitness.values == individuals[j].fitness.values:
                return True
    return False


def _generate_coeff_convex_hull_recursive(
    amount_in_lin_comb: int,
    count_unique_values: int,
    level: int = 1,
    prev_m: tuple[int] = None,
    prev_coeff: tuple[float] = None,
) -> list[tuple[float]]:
    """The recursive procedure generates coefficients for the convex hull.

    --------------------
    Args:
         amount_in_lin_comb: The number of coefficients in the convex hull.
         count_unique_values: The amount of the unique values of each coefficient in the convex hull.
                                For example, if 'amount_unique_values' = 2, then the first coefficient is {0, 1},
                                if 'amount_unique_values' = 3, then it is {0, 0.5, 1}.
                                Similarly for the rest.
         level: Recursive level.
         prev_m: The acceptable multipliers of the step in the previous level.
         prev_coeff: The acceptable coefficients in the previous level.

    --------------------
    Returns:
         The list of tuples. Each tuple is coefficients for the convex hull.

    """
    if prev_coeff is None:
        prev_coeff = tuple()

    if level == amount_in_lin_comb:
        return [prev_coeff + (1 - sum(prev_coeff),)]

    vector_of_coeff = []

    if prev_m is None:
        prev_m = tuple()

    step = 1 / (count_unique_values - 1)

    for i in range(count_unique_values - sum(prev_m)):
        coeff = i * step
        vector_of_coeff.extend(
            _generate_coeff_convex_hull_recursive(
                amount_in_lin_comb,
                count_unique_values,
                level + 1,
                (i,) + prev_m,
                (coeff,) + prev_coeff,
            )
        )

    return vector_of_coeff


def generate_coeff_convex_hull(
    amount_in_lin_comb: int, amount_unique_values: int
) -> list[tuple[float]]:
    """The procedure generates coefficients for the convex hull.

    The algorithm described in the article:
        Das, Indraneel & Dennis, J. (2000).
        Normal-Boundary Intersection:
        A New Method for Generating the Pareto Surface in Nonlinear Multicriteria Optimization Problems.
        SIAM Journal on Optimization. 8. . 10.1137/S1052623496307510.


    --------------------
    Args:
         amount_in_lin_comb: The number of coefficients in the convex hull.
         count_unique_values: The amount of the unique values of each coefficient in the convex hull.
                                For example, if 'amount_unique_values' = 2, then the first coefficient is {0, 1},
                                if 'amount_unique_values' = 3, then it is {0, 0.5, 1}.
                                Similarly for the rest.

    --------------------
    Returns:
         The list of tuples. Each tuple is coefficients for the convex hull.

    """

    assert amount_in_lin_comb > 0, "'amount_in_lin_comb' must be > 0."
    assert amount_unique_values > 1, "'amount_unique_values' must be > 1."

    return _generate_coeff_convex_hull_recursive(
        amount_in_lin_comb, amount_unique_values
    )


def asf(fitness: np.ndarray, weights: np.ndarray) -> float:
    """Achievement scalarizing function. See NSGA-3 algorithm."""
    return (fitness / weights).max()


def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


def make_gif_from_history(history, path):
    images = []
    for i in range(len(history)):
        fig = plt.figure(figsize=(7, 7))
        if len(history[i][0]) == 3:
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                history[i][:, 0],
                history[i][:, 1],
                history[i][:, 2],
                color="blue",
                alpha=0.5,
            )
            ax.set_zlabel("Objective 3")
        else:
            ax = fig.add_subplot(111)
            ax.scatter(
                history[i][:, 0],
                history[i][:, 1],
                color="blue",
                alpha=0.5,
            )
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.set_title("Pareto Front")
        image_buf = io.BytesIO()
        plt.savefig(image_buf, format="png")
        image = PIL.Image.open(image_buf)
        images.append(image)
        plt.close(fig)

    imageio.mimsave(path, images, duration=0.5)
