import numpy as np
from deap import algorithms, tools

from .printer_utils import *
from . import exploration_params, visualization
from .printer_objectives import *
from .printer_nsga2 import PrinterNSGA2


class PrinterNSGA3(PrinterNSGA2):
    def __init__(self):
        super().__init__()
        self.num_divisions = exploration_params.num_divisions

    def select(self, individuals, k):
        chosen = tools.selNSGA3(
            individuals,
            k,
            tools.uniform_reference_points(
                nobj=self.num_objectives, p=self.num_divisions
            ),
            nd=self.nd,
        )
        return chosen
