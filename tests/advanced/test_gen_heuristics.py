import unittest

# just to get cpu count
import multiprocessing
cpu_count = int(multiprocessing.cpu_count()/2)

from steinerpy.library.pipeline import GenerateHeuristics
from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.library.graphs.parser import DataParser
import steinerpy.config as cfg

# deterministic behavior
import random
random.seed(123)

class TestGenerateHeuristics(unittest.TestCase):

    def setUp(self):
        self.old_setting = cfg.Algorithm.sstar_heuristic_type
        cfg.Algorithm.sstar_heuristic_type = "diagonal_nonuniform"

    def tearDown(self):
        cfg.Algorithm.sstar_heuristic_type = self.old_setting  

    @unittest.skip("SKIPPING")
    def test_generate_all_shortest_pairs_distance(self):
        # Spec out our squareGrid
        minX = 0			# [m]
        maxX = 5         
        minY = 0
        maxY = 5
        grid = None         # pre-existing 2d numpy array?
        grid_size = 1       # grid fineness[m]
        grid_dim = [minX, maxX, minY, maxY]
        n_type = 8           # neighbor type

        # Create a squareGrid using GraphFactory
        graph = GraphFactory.create_graph("SquareGrid", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type)      

        # generate the heuristic tables
        gh = GenerateHeuristics.get_heuristics(graph, processes=cpu_count)
        
        # try retreiving some lower_bound values
        val = GenerateHeuristics.retrieve_heuristic_value(gh, (0,0), (4,5))
        
        print("\n", val)

    @unittest.skip("skip for now")
    def test_generate_landmarks(self):
        """Generate heuristics using low-level api"""
        # Spec out our squareGrid
        minX = 0			# [m]
        maxX = 15      
        minY = 0
        maxY = 15
        grid = None         # pre-existing 2d numpy array?
        grid_size = 1       # grid fineness[m]
        grid_dim = [minX, maxX, minY, maxY]
        n_type = 8           # neighbor type

        # Create a squareGrid using GraphFactory
        graph = GraphFactory.create_graph("SquareGrid", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type)      

        # generate heuristics given a graph    
        gh = GenerateHeuristics.get_heuristics(graph, processes=cpu_count)
        
        # try retreiving some lower_bound values
        val = GenerateHeuristics.retrieve_heuristic_value(gh, (0,0), (4,5))
        
        print("\n",val)

    @unittest.skip("Works but very slow as a unit test")
    def test_generate_landmarks_grid2d(self):
        """try generating landmarks for a 2d grid problem
        using the high level entry point
        
        """

        from steinerpy.environment import EnvType, EnvLoader 
        # name = "sc/WheelofWar.map"
        # graph = EnvLoader.load(EnvType.GRID_2D, name)

        # # if .sqlite in 
        # res = GenerateHeuristics.gen_and_save_results(graph, save_path="./test_wheel_of_war_h.sqlite")

        # # to load an sqlite file
        # # from sqlitedict import SqliteDict
        # # mydict = SqliteDict(""./test_wheel_of_war_h.sqlite", autocommit=True)

        pass
        # name = "den312d.map"
        name = "empty-48-48.map"
        graph = EnvLoader.load(EnvType.MAPF, name)
        # res = GenerateHeuristics.gen_and_save_results(graph, save_path="./h_test_den312d.sqlite")
        res = GenerateHeuristics.gen_and_save_results(graph, save_path="./h_test_empty-48-48.sqlite")




    # def test_manual_parallel_dijkstra(self):
    #     # Spec out our squareGrid
    #     minX = 0			# [m]
    #     maxX = 15      
    #     minY = 0
    #     maxY = 15
    #     grid = None         # pre-existing 2d numpy array?
    #     grid_size = 1       # grid fineness[m]
    #     grid_dim = [minX, maxX, minY, maxY]
    #     n_type = 8           # neighbor type

    #     # Create a squareGrid using GraphFactory
    #     graph = GraphFactory.create_graph("SquareGrid", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type)      


    

    # # WARNING VERY SLOW
    # def test_generate_landmarks_on_large_graph_orz900d(self):
    #     map_name = "orz900d.map"
    #     file_to_load = cfg.data_dir + "/mapf/" + map_name
    #     graph = DataParser.parse(file_to_load, dataset_type="mapf")

    #     gh = GenerateHeuristics.get_heuristics(graph)
    #     pass

if __name__ == "__main__":
    unittest.main(verbosity=3) 