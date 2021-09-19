import unittest

import steinerpy.config as cfg
cfg.Animation.visualize = False
cfg.Algorithm.sstar_heuristic_type = "diagonal_nonuniform"

from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.algorithms import Unmerged

class TestSteinerUnmerged(unittest.TestCase):

    def test_astar(self):

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

        # Define terminals
        terminals = [(-10, -12), (-3, 10), (10, -7), (13, 6), (0, 3)]
        # terminals = [(-15, -15), (15, 15)]
  
        # Create object
        ao = Unmerged(graph, terminals)

        # test comps type
        self.assertIsInstance(ao.comps, dict)

        # run algorithm
        self.assertTrue(ao.run_algorithm())

        self.assertTrue(isinstance(ao.return_solutions(), dict))

        self.assertTrue(len(ao.return_solutions()['sol'])>0)
        self.assertTrue(len(ao.return_solutions()['path'])>0)
        self.assertTrue(len(ao.return_solutions()['dist'])>0)

    def test_context(self):
        ''' test contextualizer '''
        print("wip, context=Context()")

if __name__ == "__main__":
    unittest.main()