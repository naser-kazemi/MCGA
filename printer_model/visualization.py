
"""
@Author Sebastian Cucerca
@Created 08/03/2023
"""


import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import numpy.typing as npt


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
    plt.savefig('%s/%s.png' % (file_path, file_name))

    #
    plt.clf()

    #
    plt.close()
