import numpy as np
import argparse as ap
import os
import imageio
import random


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

    r = np.sqrt(x1 ** 2 + x2 ** 2)
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
    r = np.sqrt(np.sum(x ** 2, axis=1))

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
    r = np.sqrt(np.sum(x ** 2))

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
    sin_prod = np.prod(np.sin(theta[1:]), axis=1)

    x[0] = r * np.cos(theta[0]) * sin_prod
    x[1] = r * np.sin(theta[0]) * sin_prod

    for i in range(2, len(x)):
        sin_prod = sin_prod / np.sin(theta[i - 1])
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
