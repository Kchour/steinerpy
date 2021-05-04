import unittest
import steinerpy.config as cfg
cfg.Animation.visualize = False
cfg.Algorithm.sstar_heuristic_type = "diagonal_nonuniform"

from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.algorithms.kruskal import Kruskal

class TestSteinerKruskal(unittest.TestCase):

    def test_kruskal(self):

        # Spec out our squareGrid
        minX = -15			# [m]
        maxX = 15           
        minY = -15
        maxY = 15
        grid = None         # pre-existing 2d numpy array?
        grid_size = 1       # grid fineness[m]
        grid_dim = [minX, maxX, minY, maxY]
        n_type = 8           # neighbor type

        # obstacles?
        # obstacles = [(0,0), (8,9), (5, 5), (11, 9), (2, 0), (4,0), (1,3)]
        obstacles = [(2,y) for y in range(-15,11)]
        obstacles.extend([(x,11) for x in range(-5,5)])
        # obstacles = []

        # Create a squareGrid using GraphFactory
        graph = GraphFactory.create_graph("SquareGrid", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type, obstacles=obstacles)      

        # Define terminals
        terminals = [(-10, -12), (-3, 10), (10, -7), (13, 6), (0, 3)]

        # Create Astar object
        ao = Kruskal(graph, terminals)

        # run algorithm
        self.assertTrue(ao.run_algorithm())

        # Get solution
        self.assertTrue(isinstance(ao.return_solutions(), dict))

        # make sure solution is non-empty
        self.assertTrue(len(ao.return_solutions()['sol'])>0)
        self.assertTrue(len(ao.return_solutions()['path'])>0)
        self.assertTrue(len(ao.return_solutions()['dist'])>0)
        

    def test_context(self):
        ''' test contextualizer '''
        print("wip, context=Context()")

if __name__ == "__main__":
    unittest.main()