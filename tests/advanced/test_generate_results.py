import unittest
import os
import sys

import steinerpy.config as cfg
cfg.Algorithm.sstar_heuristic_type = "diagonal_nonuniform"
cfg.Animation.visualize = False

from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.library.pipeline.r2generate_results import GenerateResultsMulti, GenerateResults

# Create square grid using GraphFactory
minX = -15			# [m]
maxX = 15   
minY = -15
maxY = 15
grid = None         # pre-existing 2d numpy array?
grid_size = 1       # grid fineness[m]
grid_dim = [minX, maxX, minY, maxY]
n_type = 8           # neighbor type

# Create a squareGrid using GraphFactory
sq = GraphFactory.create_graph("SquareGrid", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type)      

class TestGenerateResults(unittest.TestCase):

    def test_generate_results_multi(self):
        load_directory = os.path.dirname(__file__)
        baseline_filename = 'baseline_test_multi.pkl'
        gs = GenerateResultsMulti(sq, load_directory, baseline_filename)
        res = gs.run_func()

    def test_generate_results_single(self):
        """Runs 5 instances of 5 terminals """
        load_directory = os.path.dirname(__file__)
        baseline_filename = 'baseline_test_single.pkl'
        gs = GenerateResults(sq, load_directory, baseline_filename)
        res = gs.run_func()

    def test_generate_results_using_multi_output_but_sequentially(self):
        """Runs 100 instances of 5 terminals, test using sequentially to compare with multi """
        load_directory = os.path.dirname(__file__)
        baseline_filename = 'baseline_test_multi.pkl'
        gs = GenerateResults(sq, load_directory, baseline_filename)
        res = gs.run_func()

if __name__ == "__main__":
    unittest.main()