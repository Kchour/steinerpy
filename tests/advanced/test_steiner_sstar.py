import unittest
        
import steinerpy.config as cfg

# cfg.Animation.visualize = True
# cfg.Algorithm.sstar_heuristic_type = "diagonal_nonuniform"
# cfg.Algorithm.sstar_heuristic_type = "zero"

# change logging type test
# cfg.Misc.log_conf["handlers"]["console"]["formatter"] = "default"

# change console level
# cfg.Misc.log_conf["handlers"]["console"]["level"] = "DEBUG"
# reload log conf
# cfg.reload_log_conf()

from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.algorithms import SstarHS, SstarBS, SstarMM, SstarMM0, Unmerged

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

# Define some terminals
# terminals = [(-10, -12), (-3, 10), (10, -7), (13, 6), (0, 3)]
# terminals = [(-10, -10), (-10, 10), (10, -10), (10, 10)]
# terminals = [(-8, -4), (0, -13), (1, -5), (-15, 2), (-1, -7), (-11, 12), 
# (11, 5), (-10, -5), (7, 6), (0, 4), (-8, 11), (12, -10), (1, 1), (-6, 1),
# (-1, 11), (-3, 1), (-12, -13), (-14, -1), (-13, -12), (14, 2), (15, -10), 
# (2, 11), (5, -8), (12, 8), (15, -8), (13, 13), (0, 14), (3, 11), (-12, 0), 
# (8, 9), (-4, 6), (1, -11), (-1, 1), (0, -12), (-1, -2), (12, -3), (-6, 13)]
# terminals = [(-8, -4), (0, -13), (1, -5), (-15, 2), (-1, -7), (-11, 12), 
# (11, 5), (-10, -5), (7, 6), (0, 4), (-8, 11), (12, -10), (1, 1)]

terminals = [(-8, -4), (0, -13), (1, -5), (10, 10),(-11, 12)]

class TestSteinerSstar(unittest.TestCase):
    
    def setUp(self):
        self.old_setting = cfg.Algorithm.sstar_heuristic_type
        cfg.Algorithm.sstar_heuristic_type = "diagonal_nonuniform"
        cfg.Animation.visualize = False

    def tearDown(self):
        cfg.Algorithm.sstar_heuristic_type = self.old_setting  

    def test_returned_tree_values(self):
        """Test algorithm returned results"""

        #Store Tree values
        dist = []

        ao = SstarBS(graph, terminals)
        self.assertIsInstance(ao.comps, dict)
        self.assertTrue(ao.run_algorithm())
        dist.append(sum(ao.return_solutions()['dist']))

        # MM
        ao = SstarMM(graph, terminals)
        self.assertIsInstance(ao.comps, dict)
        self.assertTrue(ao.run_algorithm())
        dist.append(sum(ao.return_solutions()['dist']))

        # HS
        ao = SstarHS(graph, terminals)
        self.assertIsInstance(ao.comps, dict)
        self.assertTrue(ao.run_algorithm())
        dist.append(sum(ao.return_solutions()['dist']))

        # MM0
        ao = SstarMM0(graph, terminals)
        self.assertIsInstance(ao.comps, dict)
        self.assertTrue(ao.run_algorithm())
        dist.append(sum(ao.return_solutions()['dist']))

        # unmerged
        ao = Unmerged(graph, terminals)
        self.assertIsInstance(ao.comps, dict)
        self.assertTrue(ao.run_algorithm())
        dist.append(sum(ao.return_solutions()['dist']))

        # make sure tree values are the same
        # wuthin margin of floating point error
        self.assertTrue(all([abs(dist[0] - ele)<1e-6 for ele in dist]))
 
if __name__ == "__main__":
    unittest.main()