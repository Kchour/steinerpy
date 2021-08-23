import unittest
import os
import steinerpy.config as cfg
cfg.Animation.visualize = False

from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.library.pipeline.r1generate_baseline import GenerateBaseLine

# TO KEEP THE TERMINALS THE SAME
import random 
random.seed(456)

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

class TestGenerateBaseline(unittest.TestCase):

    def test_base_generate(self):
        # Create GenerateBaseLine instance
        save_path = os.path.join(cwd, "baseline_test_single.pkl")
        gen_bs = GenerateBaseLine(graph=sq, save_path=save_path, file_behavior="OVERWRITE", cache_dir=None)
        # Generate random instances
        gen_bs.randomly_generate_instances(num_of_inst=5, num_of_terms=5)
        
        # get instances
        instances = gen_bs.instances
        print(instances)

        # run and store solution
        res1 = gen_bs.run()

        # now try seeing if FileExistError is raised
        # get previous instances
  
        gen_bs = GenerateBaseLine(graph=sq, save_path=save_path, file_behavior="HALT", cache_dir=None)
        # specify instances from before
        gen_bs.input_specifed_instances(instances)
        with self.assertRaises(FileExistsError):
            gen_bs.run()

        # Now check if specified instances will be correctly inputted
        gen_bs = GenerateBaseLine(graph=sq, save_path=save_path, file_behavior="OVERWRITE", cache_dir=None)
        # specify instances from before
        gen_bs.input_specifed_instances(instances)
        res2 = gen_bs.run()
        # see if the mst values are equal
        self.assertEqual(sum(res1['solution'][0]['dist']), sum(res2['solution'][0]['dist']))
        self.assertEqual(sum(res1['solution'][1]['dist']), sum(res2['solution'][1]['dist']))

        # now don't specify save path, it should not generate any file 
        gen_bs = GenerateBaseLine(graph=sq)

    # @unittest.skip("Landlock detection not implemented")
    # def test_detect_landlocked_areas(self):
    #     # define obstacles to landlock an area
    #     obstacles = []
    #     obstacles.extend((0, y) for y in range(6))
    #     obstacles.extend((x, 5) for x in range(6))
    #     obstacles.extend((5, y) for y in range(5,-1,-1))
    #     obstacles.extend((x, 0) for x in range(5,-1,-1))
    #     # Add obstacles to graph
    #     sq_copy = sq
    #     sq_copy.set_obstacles(obstacles)

    #     # number of terminals
    #     num_terminals = 100
    #     # number of instances
    #     num_instances = 5
    #     # directory to save baseline file
    #     save_directory = os.path.dirname(__file__)
    #     filename = 'baseline_test_multi.pkl'
    #     gm = GenerateBaseLineMulti(sq_copy, num_terminals, num_instances, save_directory, filename, file_behavior="OVERWRITE")
    #     res = gm.run_func()
    #     print("")
  
if __name__ == "__main__":
    unittest.main()