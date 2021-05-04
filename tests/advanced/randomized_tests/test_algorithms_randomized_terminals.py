import unittest
import os
import sys
import random

import steinerpy.config as cfg
cfg.Algorithm.sstar_heuristic_type = "diagonal_nonuniform"
cfg.Animation.visualize = False

from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.library.pipeline.r2generate_results import GenerateResultsMulti

# Create square grid using GraphFactory
minX = -25			# [m]
maxX = 25   
minY = -25
maxY = 25
grid = None         # pre-existing 2d numpy array?
grid_size = 1       # grid fineness[m]
grid_dim = [minX, maxX, minY, maxY]
n_type = 8           # neighbor type

# Create a squareGrid using GraphFactory
sq = GraphFactory.create_graph("SquareGrid", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type)      

# Random terminal instances to run
M = 1000
# Limit number of terminals per instance
N = 25
terminal_list = []
temp = set()
while len(terminal_list) < M:
    x = random.randint(minX, maxX)
    y = random.randint(minY, maxY)
    if len(temp) < N:
        temp.add((x,y))
    else:
        terminal_list.append(list(temp))
        temp = set()


class TestGenerateResults(unittest.TestCase):

    def test_generate_results_multi(self):
        # load_directory = os.path.dirname(__file__)
        # baseline_filename = 'baseline_test_multi.pkl'
        # gs = GenerateResultsMulti(sq, load_directory, baseline_filename)
        # res = gs.run_func()
        pass

if __name__ == "__main__":
    unittest.main()