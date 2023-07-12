import numpy as np
from population import Population
from member import Member


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


if __name__ == "__main__":
    # create a population of 10 members with 5 objectives
    # population = Population([])
    # for _ in range(10):
    #     objectives = np.random.rand(5)
    #     population.append(Member(objectives))
    #
    # print("========================")
    # print("Unsorted population:")
    # print(population)
    # print("========================\n\n")
    #
    # # sort the population
    # sorted_population = sort_population_by_value(population)
    # print("========================")
    # print("Sorted population:")
    # print(sorted_population)
    # print("========================\n\n")
    pass
