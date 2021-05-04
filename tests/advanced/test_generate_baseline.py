import unittest
import os
import steinerpy.config as cfg
cfg.Animation.visualize = False

from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.library.pipeline.r1generate_baseline import GenerateBaseLineMulti, GenerateBaseLine


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

class TestGenerateBaseline(unittest.TestCase):

    def test_base_generate_multi(self):
        # number of terminals
        num_terminals = 100
        # number of instances
        num_instances = 2
        # directory to save baseline file
        # save_directory = cfg.results_dir+"/tests/"
        save_directory =  os.path.dirname(__file__)
        filename = 'baseline_test_multi.pkl'
        # cache should be based on the map!
        cachename = 'baseline_test_multi.pkl' 
        gm = GenerateBaseLineMulti(sq, num_terminals, num_instances, save_directory, cachename, filename, file_behavior="OVERWRITE")
        res = gm.run_func()
        print("")

    @unittest.skip("Landlock detection not implemented")
    def test_detect_landlocked_areas(self):
        # define obstacles to landlock an area
        obstacles = []
        obstacles.extend((0, y) for y in range(6))
        obstacles.extend((x, 5) for x in range(6))
        obstacles.extend((5, y) for y in range(5,-1,-1))
        obstacles.extend((x, 0) for x in range(5,-1,-1))
        # Add obstacles to graph
        sq_copy = sq
        sq_copy.set_obstacles(obstacles)

        # number of terminals
        num_terminals = 100
        # number of instances
        num_instances = 5
        # directory to save baseline file
        save_directory = os.path.dirname(__file__)
        filename = 'baseline_test_multi.pkl'
        gm = GenerateBaseLineMulti(sq_copy, num_terminals, num_instances, save_directory, filename, file_behavior="OVERWRITE")
        res = gm.run_func()
        print("")
  
if __name__ == "__main__":
    unittest.main()