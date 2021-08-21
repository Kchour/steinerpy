import unittest
import os
import sys
import random
import pickle

import steinerpy.config as cfg
cfg.Algorithm.sstar_heuristic_type = "diagonal_nonuniform"
cfg.Animation.visualize = False

from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.library.graphs.parser import DataParser
from steinerpy.library.pipeline import GenerateResultsMulti, GenerateBaseLineMulti
from steinerpy.algorithms import SstarHS, SstarHS0, SstarBS, SstarMM, SstarMM0, Kruskal

# # optionally use context to choose algorithms
# from steinerpy import context

# set seed if desired
random.seed(123)

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
sq.name = "square"

# Load other maps
mapf_maze = DataParser.parse(os.path.join(cfg.data_dir,"mapf", "maze-32-32-2.map"), dataset_type="mapf")
mapf_maze.name = "maze"
mapf_den = DataParser.parse(os.path.join(cfg.data_dir,"mapf", "den312d.map"), dataset_type="mapf")
mapf_den.name = "den"

test_maps = [sq]

def generate_random_terminals(_map, num_of_inst, num_of_terms):
    """Generate random unique terminals which do not coincide with obstacles

    """
    terminal_list = []
    temp = set()
    minX, maxX, minY, maxY = _map.grid_dim

    # keep going until we have enough terminals
    while len(terminal_list) < num_of_inst:
        # randomly sample
        x = random.randint(minX, maxX)
        y = random.randint(minY, maxY)
        if len(temp) < num_of_terms:
            # make sure terminal is not an obstacle in current map
            if _map.not_obstacles((x,y)):
                temp.add((x,y))
        else:
            terminal_list.append(list(temp))
            temp = set()

    return terminal_list


# get current working directory where this script is being run
working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")

class TestGenerateAndCompareResults(unittest.TestCase):

    def test_generate_results_multi(self):
        num_of_inst = 1000
        num_of_terms = 30
        for _map in test_maps:
            # Generate a list of list of random terminals
            term_inst = generate_random_terminals(_map, num_of_inst, num_of_terms)
            print("GEN BASELINE")
            gen_bm = GenerateBaseLineMulti(_map, num_of_terms, num_of_inst, working_dir,"".join(("baseline_",_map.name,".pkl")), None, file_behavior="SKIP")
            gen_bm.run_func()
            
            # Generate results
            print("GEN RESULTS")
            gen_rm = GenerateResultsMulti(_map, working_dir, "".join(("baseline_",_map.name,".pkl")), working_dir, file_behavior="SKIP")
            gen_rm.run_func(["S*-BS", "S*-HS", "S*-MM", "S*-MM0"])

            # process them
            
            # compare them
        # load_directory = os.path.dirname(__file__)
        # baseline_filename = 'baseline_test_multi.pkl'
        # gs = GenerateResultsMulti(sq, load_directory, baseline_filename)
        # res = gs.run_func()
        pass

if __name__ == "__main__":
    unittest.main()