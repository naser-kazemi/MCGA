import numpy as np
from population import Population
from member import Member
import argparse as ap
import os


def index_of(a: list, value):
    """
    Function to get the first index of a value in a list
    """
    for i in range(len(a)):
        if a[i] == value:
            return i
    return -1


def sort_population_by_value(population: Population) -> Population:
    """
    Sort the population in ascending order by the objective values
    :param population: The population to sort
    :return: The sorted population
    """
    population_members = population.population
    sorted_members = sorted(population_members)
    return Population(sorted_members)


def sort_front_by_value(front: set) -> Population:
    """
    Sort the front in ascending order by the objective values
    :param front: The front to sort
    :return: The sorted front
    """
    front_population = Population(list(front))
    return sort_population_by_value(front_population)


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
