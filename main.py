from nsga2 import NSGA2
from utils import *


def main():
    args = create_parser()

    # create a new NSGA2 instance
    # create two objectives
    obj1 = lambda x: x[0] ** 2
    obj2 = lambda x: (x[0] - 2) ** 2
    objectives = [obj1, obj2]
    nsga2 = NSGA2(args.population_size, args.num_variables, args.num_objectives, objectives,
                  num_generations=args.num_generations, tournament_size=args.tournament_size,
                  eta_crossover=args.eta_crossover, eta_mutation=args.eta_mutation,
                  crossover_probability=args.crossover_probability)

    print("initial population size: ", len(nsga2.population))

    # run the algorithm
    nsga2.run()

    # print(nsga2.population)


if __name__ == "__main__":
    main()
