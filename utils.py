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
    args.add_argument("--population-size", type=int, default=100, help="The size of the population")
    args.add_argument("--num-variables", type=int, default=2, help="The number of variables")
    args.add_argument("--num-objectives", type=int, default=2, help="The number of objectives")
    args.add_argument("--num-generations", type=int, default=100, help="The number of generations")
    args.add_argument("--tournament-size", type=int, default=2, help="The tournament size")
    args.add_argument("--eta-crossover", type=float, default=1.0, help="The eta crossover")
    args.add_argument("--eta-mutation", type=float, default=1.0, help="The eta mutation")
    args.add_argument("--crossover-probability", type=float, default=0.9, help="The crossover probability")
    args.add_argument("--output-dir", type=str, default="output", help="The output directory")
    return args.parse_args()


def create_gif(images_directory="images", output_file="output.gif"):
    """
    Create a GIF from the images in the images directory
    """

    # get the list of images in the images directory
    images_list = os.listdir(images_directory)
    # sort the list of images
    images_list.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))

    frames = []
    # read all images in the images directory and add them to the frames list
    for image in images_list:
        frames.append(imageio.v2.imread(os.path.join(images_directory, image)))

    # create the GIF
    imageio.mimsave(output_file, frames, duration=200)
