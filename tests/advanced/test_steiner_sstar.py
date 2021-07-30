import unittest

import steinerpy.config as cfg
cfg.Animation.visualize = True
cfg.Algorithm.sstar_heuristic_type = "diagonal_nonuniform"

from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.algorithms import SstarHS, SstarHS0, SstarBS

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
    
    def test_returned_tree_values(self):
        #Store Tree values
        dist = []

        # # Create Astar object
        # ao = SstarHS(graph, terminals)
        # # test comps type
        # self.assertIsInstance(ao.comps, dict)
        # # run algorithm
        # self.assertTrue(ao.run_algorithm())
        # dist.append(sum(ao.return_solutions()['dist']))

        # Test Primal Dual
        # Create Astar object
        ao = SstarBS(graph, terminals)
        # test comps type
        self.assertIsInstance(ao.comps, dict)
        # run algorithm
        self.assertTrue(ao.run_algorithm())
        dist.append(sum(ao.return_solutions()['dist']))

        # Test Dijkstra
        ao = SstarHS0(graph, terminals)
        # test comps type
        self.assertIsInstance(ao.comps, dict)
        # Run algorithm, does it return true?
        self.assertTrue(ao.run_algorithm())
        # Save dist
        dist.append(sum(ao.return_solutions()['dist']))

        # Test equivalence
        self.assertTrue(abs(max(dist) - min(dist))<=1e-9)

    # def test_sstar_dijkstra(self):

    #     # Create Astar object
    #     ao = SstarDijkstra(graph, terminals)

    #     # test comps type
    #     self.assertIsInstance(ao.comps, dict)

    #     # run algorithm
    #     self.assertTrue(ao.run_algorithm())

    # def test_sstar_astar(self):

    #     # Create Astar object
    #     ao = SstarAstar(graph, terminals)

    #     # test comps type
    #     self.assertIsInstance(ao.comps, dict)

    #     # run algorithm
    #     self.assertTrue(ao.run_algorithm())

    # def test_sstar_primaldual(self):

    #     # Create Astar object
    #     ao = SstarPrimalDual(graph, terminals)

    #     # test comps type
    #     self.assertIsInstance(ao.comps, dict)

    #     # run algorithm
    #     self.assertTrue(ao.run_algorithm())

    # def test_context(self):
    #     ''' test contextualizer '''
    #     print("wip, context=Context()")

if __name__ == "__main__":
    unittest.main()