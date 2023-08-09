from pymoo.problems import get_problem
from emoa.moop import MOOP

import matplotlib.pyplot as plt
from emoa.utils import *
from nsga2.model import NSGA2
from mcga.model import MCGA
import math


def main():
    problem_name = "dtlz1"
    problem = get_problem(problem_name)
    lower_bound = problem.xl
    upper_bound = problem.xu

    # create a new NSGA2 instance
    # create three objectives for the DTLZ1 problem
    g = lambda x: 100 * (
            (len(x) - 2)
            + sum(
        [(x_i - 0.5) ** 2 - math.cos(20 * math.pi * (x_i - 0.5)) for x_i in x[2:]]
    )
    )
    f1 = lambda x: 0.5 * x[0] * x[1] * (1 + g(x))
    f2 = lambda x: 0.5 * x[0] * (1 - x[1]) * (1 + g(x))
    f3 = lambda x: 0.5 * (1 - x[0]) * (1 + g(x))
    objectives = [f1, f2, f3]

    moop = MOOP(
        problem.n_var, objectives, problem.pareto_front(), lower_bound, upper_bound
    )

    mcga = MCGA(moop, 300, 100, 0.9, 2, 1, 1, np.pi, 20, 0.01)

    # clearing the gif_images directory
    for image in os.listdir("../gif_images"):
        os.remove(os.path.join("../gif_images", image))
    for image in os.listdir("../monte_carlo_gif_images"):
        os.remove(os.path.join("../monte_carlo_gif_images", image))

    # run the algorithm
    mcga.run()
    front = np.array(mcga.fast_non_dominated_sort(mcga.population)[0])
    front = np.array([member.objective_values for member in front])
    distance_mean, distance_std = mcga.evaluate_distance_metric()
    print("Distance metric mean: ", distance_mean)
    print("Distance metric std: ", distance_std)
    diversity = mcga.evaluate_diversity_metric()
    print("Diversity metric: ", diversity)

    # plot the results and save the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        problem.pareto_front()[:, 0],
        problem.pareto_front()[:, 1],
        problem.pareto_front()[:, 2],
        color="red",
        label="Pareto Front",
    )
    ax.scatter(front[:, 0], front[:, 1], front[:, 2], color="blue", label="MCGA")

    plt.xlabel("$f_1(x)$")
    plt.ylabel("$f_2(x)$")
    ax.set_zlabel("$f_3(x)$")
    plt.title(problem_name.upper())
    ax.view_init(30, 40)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    plt.legend()
    plt.savefig(f"images/{problem_name}_mcga.png")
    plt.close()

    # create a GIF from the gif_images in the gif_images directory
    create_gif("gif_images", f"./gifs/{problem_name}_mcga.gif")

    nsga2 = NSGA2(moop, 300, 100, 0.9, 20, 20)

    # clearing the gif_images directory
    for image in os.listdir("../gif_images"):
        os.remove(os.path.join("../gif_images", image))
    for image in os.listdir("../monte_carlo_gif_images"):
        os.remove(os.path.join("../monte_carlo_gif_images", image))

    # run the algorithm
    nsga2.run()
    front = np.array(nsga2.fast_non_dominated_sort(nsga2.population)[0])
    front = np.array([member.objective_values for member in front])
    distance_mean, distance_std = nsga2.evaluate_distance_metric()
    print("Distance metric mean: ", distance_mean)
    print("Distance metric std: ", distance_std)
    diversity = nsga2.evaluate_diversity_metric()
    print("Diversity metric: ", diversity)

    # plot the results and save the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        problem.pareto_front()[:, 0],
        problem.pareto_front()[:, 1],
        problem.pareto_front()[:, 2],
        color="red",
        label="Pareto Front",
    )
    ax.scatter(front[:, 0], front[:, 1], front[:, 2], color="blue", label="NSGA2")

    plt.xlabel("$f_1(x)$")
    plt.ylabel("$f_2(x)$")
    ax.set_zlabel("$f_3(x)$")
    plt.title(problem_name.upper())
    ax.view_init(30, 40)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    plt.legend()
    plt.savefig(f"images/{problem_name}_nsga.png")
    plt.close()

    create_gif("gif_images", f"./gifs/{problem_name}_nsga.gif")


if __name__ == "__main__":
    main()