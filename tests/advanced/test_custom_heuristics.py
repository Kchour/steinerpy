import unittest
import os
import random
import cProfile, pstats

# for profiling code
profiler = cProfile.Profile()

# Set seed if desired
random.seed(123)    # 50 terminal
# random.seed(1234)    # 2 terminal

import steinerpy.config as cfg                              # to ensure heuristic is set to "preprocess"
from steinerpy.context import Context                       # helper to load graph and run algorithms
from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.algorithms.common import Common

# Visualize things
# cfg.Animation.visualize = True

 # Spec out our squareGrid
minX = -15			# [m]
maxX = 15           
minY = -15
maxY = 15
grid = None         # pre-existing 2d numpy array?
grid_size = 1       # grid fineness[m]
grid_dim = [minX, maxX, minY, maxY]
n_type = 8           # neighbor type

# Create a squareGrid using GraphFactory
graph = GraphFactory.create_graph("SquareGrid", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type) 

# generate random unique set of terminals
T = set()
while len(T) < 50:
    x = random.randint(minX, maxX)
    y = random.randint(minY, maxY)
    T.add((x,y))

# convert back to list!
T = list(T)

class TestGenerateHeuristics(unittest.TestCase):

    def my_cust_h_func(self, n, goal):
        # print(n, goal)
        return 0

    def test_load_heuristics_from_disk_run_sstar(self):

        # change heuristic setting
        cfg.Algorithm.sstar_heuristic_type = "custom"    

        # bind custom_heuristics function
        Common.custom_heuristics = self.my_cust_h_func

        # create context handler and run S*-mm
        context = Context(graph, T)

        # profiler enable
        profiler.enable()

        # Now run
        context.run("S*-MM")

        # disable profiler
        profiler.disable()

        # store results
        results = context.return_solutions()   

        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()

        print(results)

if __name__ == "__main__":
    unittest.main() 