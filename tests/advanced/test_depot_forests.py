import unittest
import steinerpy.config as cfg
cfg.Animation.visualize = False
cfg.Algorithm.sstar_heuristic_type = "diagonal_nonuniform"

from steinerpy.context import Context
from steinerpy.library.graphs.graph import GraphFactory, SquareGridDepot
from steinerpy.algorithms import SstarHS, SstarHS0, SstarBS
from steinerpy.library.graphs.parser import DataParser

class TestSteinerSstar(unittest.TestCase):
    
    def test_returned_tree_values(self):
        minX = -15			# [m]
        maxX = 15           
        minY = -15
        maxY = 15
        grid = None         # pre-existing 2d numpy array?
        grid_size = 1       # grid fineness[m]
        grid_dim = [minX, maxX, minY, maxY]
        n_type = 8           # neighbor type

        # DEPOTS MUST BE A SUBSET OF TERMINALS!
        depots = [(-8,-4), (12,-3), (-11,12), (1,-11), (-4,6)]

        terminals = [(-8, -4), (0, -13), (1, -5), (-15, 2), (-1, -7), (-11, 12), 
        (11, 5), (-10, -5), (7, 6), (0, 4), (-8, 11), (12, -10), (1, 1), (-6, 1),
        (-1, 11), (-3, 1), (-12, -13), (-14, -1), (-13, -12), (14, 2), (15, -10), 
        (2, 11), (5, -8), (12, 8), (15, -8), (13, 13), (0, 14), (3, 11), (-12, 0), 
        (8, 9), (-4, 6), (1, -11), (-1, 1), (0, -12), (-1, -2), (12, -3), (-6, 13)]
        # Create a squareGrid using GraphFactory
        graph = GraphFactory.create_graph("SquareGridDepot", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type, depots=depots)

        #Store Tree values
        dist = []

        # # Test Dijkstra
        # ao = SstarDijkstra(graph, terminals)
        # # test comps type
        # self.assertIsInstance(ao.comps, dict)
        # # Run algorithm, does it return true?
        # self.assertTrue(ao.run_algorithm())
        # # Save dist
        # dist.append(sum(ao.return_solutions()['dist']))

        # Create Astar object
        ao = SstarHS(graph, terminals)
        # test comps type
        self.assertIsInstance(ao.comps, dict)
        # run algorithm
        self.assertTrue(ao.run_algorithm())
        dist.append(sum(ao.return_solutions()['dist']))

        # Test Primal Dual
        # Create Astar object
        ao = SstarBS(graph, terminals)
        # test comps type
        self.assertIsInstance(ao.comps, dict)
        # run algorithm
        self.assertTrue(ao.run_algorithm())
        dist.append(sum(ao.return_solutions()['dist']))

        # Test equivalence
        self.assertTrue(abs(max(dist) - min(dist))<=1e-9)

    def test_null_depot_error(self):
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
        self.assertRaises(ValueError, GraphFactory.create_graph, "SquareGridDepot", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type)


    def test_sstar_primaldual_with_depots(self):

        # Spec out our squareGrid
        minX = -15			# [m]
        maxX = 15           
        minY = -15
        maxY = 15
        grid = None         # pre-existing 2d numpy array?
        grid_size = 1       # grid fineness[m]
        grid_dim = [minX, maxX, minY, maxY]
        n_type = 8           # neighbor type

        # Define some depots
        # depots = [(-10,-12), (0,3)]
        depots = [(-15,-15), (-15,15)]

        # Create a squareGridDepot using GraphFactory
        graph = GraphFactory.create_graph("SquareGridDepot", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type, depots=depots)      

        # Define some terminals
        # terminals = [(-10, -12), (-3, 10), (10, -7), (13, 6), (0, 3)]
        terminals = [(-15,-15), (-15,15), (15,15), (15,-15)]

        # Create Astar object
        ao = SstarBS(graph, terminals)

        # test comps type
        self.assertIsInstance(ao.comps, dict)

        # run algorithm
        self.assertTrue(ao.run_algorithm())    

    def test_sstar_astar_with_depots(self):

        # Spec out our squareGrid
        minX = -15			# [m]
        maxX = 15           
        minY = -15
        maxY = 15
        grid = None         # pre-existing 2d numpy array?
        grid_size = 1       # grid fineness[m]
        grid_dim = [minX, maxX, minY, maxY]
        n_type = 8           # neighbor type

        # Define some depots
        # depots = [(-10,-12), (0,3)]
        # depots = [(-15,-15), (-15,15)]
        # depots = [(-8,-4), (1,-11), (12,-10), (-15, 2), (12,-3), (-11,12)]
        depots = [(-8,-4), (12,-3), (-11,12), (1,-11), (-4,6)]
        # depots = [(-8,-4), (-11,12), (12,-10)]

        # Create a squareGridDepot using GraphFactory
        graph = GraphFactory.create_graph("SquareGridDepot", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type, depots=depots)      

        # Define some terminals
        # terminals = [(-10, -12), (-3, 10), (10, -7), (13, 6), (0, 3)]
        # terminals = [(-15,-15), (-15,15), (15,15), (15,-15)]
        terminals = [(-8, -4), (0, -13), (1, -5), (-15, 2), (-1, -7), (-11, 12), 
        (11, 5), (-10, -5), (7, 6), (0, 4), (-8, 11), (12, -10), (1, 1), (-6, 1),
        (-1, 11), (-3, 1), (-12, -13), (-14, -1), (-13, -12), (14, 2), (15, -10), 
        (2, 11), (5, -8), (12, 8), (15, -8), (13, 13), (0, 14), (3, 11), (-12, 0), 
        (8, 9), (-4, 6), (1, -11), (-1, 1), (0, -12), (-1, -2), (12, -3), (-6, 13)]
        # terminals = [(-8, -4), (0, -13), (1, -5), (-15, 2), (-1, -7), (-11, 12)]

        # Create Astar object
        ao = SstarHS(graph, terminals)

        # test comps type
        self.assertIsInstance(ao.comps, dict)

        # run algorithm
        self.assertTrue(ao.run_algorithm())

    def test_kruskal_with_depots(self):

        # Spec out our squareGrid
        minX = -15			# [m]
        maxX = 15           
        minY = -15
        maxY = 15
        grid = None         # pre-existing 2d numpy array?
        grid_size = 1       # grid fineness[m]
        grid_dim = [minX, maxX, minY, maxY]
        n_type = 8           # neighbor type
        # neighbor type

        # Define some depots
        # depots = [(-8,-4), (1,-11), (12,-10)]
        # depots = [(-15,-15), (-15,15)]
        # depots = [(27, 11), (45, 22), (62, 31), (13, 26), (35, 41)]
        depots = [(-8,-4), (12,-3), (-11,12), (1,-11), (-4,6)]

        # Create a squareGridDepot using GraphFactory
        graph = GraphFactory.create_graph("SquareGridDepot", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type, depots=depots)      

        # Define some terminals
        # terminals = [(-10, -12), (-3, 10), (10, -7), (13, 6), (0, 3)]
        # terminals = [(-15,-15), (-15,15), (15,15), (15,-15)]
        terminals = [(-8, -4), (0, -13), (1, -5), (-15, 2), (-1, -7), (-11, 12), 
        (11, 5), (-10, -5), (7, 6), (0, 4), (-8, 11), (12, -10), (1, 1), (-6, 1),
        (-1, 11), (-3, 1), (-12, -13), (-14, -1), (-13, -12), (14, 2), (15, -10), 
        (2, 11), (5, -8), (12, 8), (15, -8), (13, 13), (0, 14), (3, 11), (-12, 0), 
        (8, 9), (-4, 6), (1, -11), (-1, 1), (0, -12), (-1, -2), (12, -3), (-6, 13)]
        # terminals = [(62, 15), (63, 14), (3, 52), (35, 41), (59, 36), (7, 42), (28, 26), (45, 22), (62, 31), (28, 46), (23, 44), (5, 60), (10, 35), (61, 60), (36, 50), (57, 59), (19, 60), (59, 46), (53, 23), (63, 44), (44, 36), (59, 4), (33, 9), (44, 55), (63, 10), (45, 54), (49, 51), (13, 26), (43, 29), (0, 49), (35, 51), (27, 11), (25, 47), (38, 44), (49, 36), (23, 63), (55, 49), (11, 31), (28, 42), (50, 46), (46, 13), (55, 34), (34, 26), (19, 17), (44, 57), (17, 1), (31, 17), (4, 58), (31, 14), (21, 55), (7, 4), (39, 51), (4, 44), (62, 25), (2, 29), (59, 18), (11, 14), (55, 13), (29, 39), (7, 33), (23, 17), (6, 62), (23, 15), (27, 32), (13, 28), (25, 52), (53, 12), (60, 31), (43, 49), (35, 2), (60, 10), (7, 13), (49, 18), (38, 36), (57, 10), (27, 52), (14, 30), (28, 55), (35, 18), (17, 3), (38, 13), (9, 37), (41, 19), (43, 37), (34, 1), (30, 28), (53, 63), (60, 1), (18, 60), (54, 35), (52, 26), (61, 51), (34, 40), (7, 45), (26, 63), (45, 29), (50, 31), (35, 42), (60, 34), (54, 29)]

        # Use context to run kruskal
        context = Context(graph, terminals)
        
        self.assertTrue(context.run('Kruskal'))
        results = context.return_solutions()

        # test comps type
        self.assertIsInstance(results, dict)

    ###############

    # def test_kruskal_with_depots_with_obstacles(self):

    #     # Spec out our squareGrid
    #     # minX = 0			# [m]
    #     # maxX = 64          
    #     # minY = 0
    #     # maxY = 64
    #     # grid = None         # pre-existing 2d numpy array?
    #     # grid_size = 1       # grid fineness[m]
    #     # grid_dim = [minX, maxX, minY, maxY]
    #     # n_type = 8           # neighbor type
    #     # neighbor type

    #     # Define some depots
    #     # depots = [(-8,-4), (1,-11), (12,-10)]
    #     # depots = [(-15,-15), (-15,15)]
    #     depots = [(27, 11), (45, 22), (62, 31), (13, 26), (35, 41)]

    #     # Create a squareGridDepot using GraphFactory
    #     # graph = GraphFactory.create_graph("SquareGridDepot", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type, depots=depots)      
    #     data_file = 'room-64-64-8.map'
    #     data_directory = cfg.data_dir + '/mapf'
    #     depots = [(27, 11), (45, 22), (62, 31), (13, 26), (35, 41)]
    #     graph = DataParser.parse(data_directory+"/"+data_file, dataset_type="mapf", depots=depots)

    #     # Define some terminals
    #     # terminals = [(-10, -12), (-3, 10), (10, -7), (13, 6), (0, 3)]
    #     # terminals = [(-15,-15), (-15,15), (15,15), (15,-15)]
    #     # terminals = [(-8, -4), (0, -13), (1, -5), (-15, 2), (-1, -7), (-11, 12), 
    #     # (11, 5), (-10, -5), (7, 6), (0, 4), (-8, 11), (12, -10), (1, 1), (-6, 1),
    #     # (-1, 11), (-3, 1), (-12, -13), (-14, -1), (-13, -12), (14, 2), (15, -10), 
    #     # (2, 11), (5, -8), (12, 8), (15, -8), (13, 13), (0, 14), (3, 11), (-12, 0), 
    #     # (8, 9), (-4, 6), (1, -11), (-1, 1), (0, -12), (-1, -2), (12, -3), (-6, 13)]
    #     terminals = [(62, 15), (63, 14), (3, 52), (35, 41), (59, 36), (7, 42), (28, 26), (45, 22), (62, 31), (28, 46), (23, 44), (5, 60), (10, 35), (61, 60), (36, 50), (57, 59), (19, 60), (59, 46), (53, 23), (63, 44), (44, 36), (59, 4), (33, 9), (44, 55), (63, 10), (45, 54), (49, 51), (13, 26), (43, 29), (0, 49), (35, 51), (27, 11), (25, 47), (38, 44), (49, 36), (23, 63), (55, 49), (11, 31), (28, 42), (50, 46), (46, 13), (55, 34), (34, 26), (19, 17), (44, 57), (17, 1), (31, 17), (4, 58), (31, 14), (21, 55), (7, 4), (39, 51), (4, 44), (62, 25), (2, 29), (59, 18), (11, 14), (55, 13), (29, 39), (7, 33), (23, 17), (6, 62), (23, 15), (27, 32), (13, 28), (25, 52), (53, 12), (60, 31), (43, 49), (35, 2), (60, 10), (7, 13), (49, 18), (38, 36), (57, 10), (27, 52), (14, 30), (28, 55), (35, 18), (17, 3), (38, 13), (9, 37), (41, 19), (43, 37), (34, 1), (30, 28), (53, 63), (60, 1), (18, 60), (54, 35), (52, 26), (61, 51), (34, 40), (7, 45), (26, 63), (45, 29), (50, 31), (35, 42), (60, 34), (54, 29)]

    #     # Use context to run kruskal
    #     context = Context(graph, terminals)
        
    #     self.assertTrue(context.run('Kruskal'))
    #     results = context.return_solutions()

    #     # test comps type
    #     self.assertIsInstance(results, dict)

if __name__ == "__main__":
    unittest.main()