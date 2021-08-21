from multiprocessing import Value
import unittest
import os
import sys
import logging

# FOR DETERMINSTIC BEHAVIOR
import random 
random.seed(456)

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

# get current working directory (the test folder)
cwd = os.path.dirname(os.path.abspath(__file__))

my_logger = logging.getLogger(__name__)

class TestGenerateResults(unittest.TestCase):

    # @unittest.skip("TESTING PURPOSES")
    def test_main_results_generate_single(self):
        save_path = os.path.join(cwd, "main_results_test_single.pkl")
        algs_to_run = ["S*-HS", "S*-BS", "S*-MM", "S*-MM0"]
        gen_mr = GenerateResults(graph=sq, save_path=save_path, file_behavior="OVERWRITE", algs_to_run=algs_to_run)

        gen_mr.randomly_generate_instances(5, 25)

        res = gen_mr.run()

        instances = res['terminals']

        for ndx in range(len(instances)):
            mst_values = []
            for alg in algs_to_run:
                mst_values.append(sum(res['solution'][alg][ndx]['dist']))

            self.assertTrue(all(mst_values[0] == ele for ele in mst_values))     
               
    def test_main_results_generate_multi(self):
        save_path = os.path.join(cwd, "main_results_test_multi.pkl")
        algs_to_run = ["S*-HS", "S*-BS", "S*-MM", "S*-MM0"]
        gen_mr = GenerateResultsMulti(graph=sq, save_path=save_path, file_behavior="OVERWRITE", algs_to_run=algs_to_run)

        gen_mr.randomly_generate_instances(5, 25)

        instances = gen_mr.instances
        print(instances)

        res = gen_mr.run()


        for ndx in range(len(instances)):
            mst_values = []
            for alg in algs_to_run:
                mst_values.append(sum(res['solution'][alg][ndx]['dist']))
                print(ndx, alg, mst_values)

            try:
                if not all(abs(mst_values[0] - ele)<1e-6  for ele in mst_values):
                    raise ValueError("Uh oh mst values don't match {} {} {}".format(algs_to_run, mst_values, instances[ndx]))
            except:
                my_logger.warn("test_main_results_generate_multi", exc_info=True)

            # self.assertTrue(all(mst_values[0] == ele for ele in mst_values))   
    # def test_generate_results_multi(self):
    #     load_directory = os.path.dirname(__file__)
    #     baseline_filename = 'baseline_test_multi.pkl'
    #     gs = GenerateResultsMulti(sq, load_directory, baseline_filename)
    #     res = gs.run_func()

    # def test_generate_results_single(self):
    #     """Runs 5 instances of 5 terminals """
    #     load_directory = os.path.dirname(__file__)
    #     baseline_filename = 'baseline_test_single.pkl'
    #     gs = GenerateResults(sq, load_directory, baseline_filename)
    #     res = gs.run_func()

    # def test_generate_results_using_multi_output_but_sequentially(self):
    #     """Runs 100 instances of 5 terminals, test using sequentially to compare with multi """
    #     load_directory = os.path.dirname(__file__)
    #     baseline_filename = 'baseline_test_multi.pkl'
    #     gs = GenerateResults(sq, load_directory, baseline_filename)
    #     res = gs.run_func()

if __name__ == "__main__":
    unittest.main()