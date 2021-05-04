import unittest
import os

from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.library.pipeline.r3process import Process
import steinerpy.config as cfg

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

class TestProcessResults(unittest.TestCase):

    def test_process_results_multicore(self):
        """Runs 20 instances of 50 terminals """
        baseline_directory = os.path.dirname(__file__)
        results_directory = os.path.dirname(__file__)
        baseline_filename = 'baseline_test_multi.pkl'
        results_filename = 'results_test_multi.pkl'
        pr = Process(baseline_directory, results_directory, baseline_filename=baseline_filename,
            results_filename=results_filename)
        pr.run_func()

    def test_process_results_sequentially(self):
        """Runs 5 instances of 5 terminals """
        baseline_directory = os.path.dirname(__file__)
        results_directory = os.path.dirname(__file__)
        baseline_filename = 'baseline_test_single.pkl'
        results_filename = 'results_test_single.pkl'
        pr = Process(baseline_directory, results_directory, baseline_filename=baseline_filename,
            results_filename=results_filename)
        pr.run_func()


if __name__ == "__main__":
    unittest.main()