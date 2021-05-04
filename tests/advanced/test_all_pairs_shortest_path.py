
import cProfile
from memory_profiler import profile
import unittest
from timeit import default_timer as timer

from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.library.search.all_pairs_shortest_path import AllPairsShortestPath


class TestAllPairsShortestPath(unittest.TestCase):
    def test_floyd_warshall_simple_slow(self):
        # Spec out our squareGrid
        minX = 0			# [m]
        maxX = 7          
        minY = 0
        maxY = 7
        grid = None         # pre-existing 2d numpy array?
        grid_size = 1       # grid fineness[m]
        grid_dim = [minX, maxX, minY, maxY]
        n_type = 8           # neighbor type

        # Create a squareGrid using GraphFactory
        graph = GraphFactory.create_graph("SquareGrid", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type)      

        #try running floyd_warshall
        fl = AllPairsShortestPath.floyd_warshall_simple_slow(graph)
        print("")

    def test_dijkstra_parallel_user_specifications(self):
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

        # Either sample a percentage of the configuration space or get a fixed number of samples
        fl_dij = AllPairsShortestPath.dijkstra_in_parallel(graph)
        #user specify sampling percentage
        fl_dij = AllPairsShortestPath.dijkstra_in_parallel(graph, random_sampling_percentage=10)

        # user specify nodes to run
        nodes = [(1, 1), (3, 0), (2, 4), (2, 2)]
        fl_dij = AllPairsShortestPath.dijkstra_in_parallel(graph, nodes=nodes)
        # 
        # user specify sampling limit
        fl_dij = AllPairsShortestPath.dijkstra_in_parallel(graph, random_sampling_limit=10)
        pass

    def test_all_pairs_shortest_path_methods(self):
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

        tstart = timer()
        fl_dij = AllPairsShortestPath.dijkstra_in_parallel(graph, flatten_results_into_pairs=True)
        fl_dij_time = timer()-tstart

        #try running floyd_warshall
        # c = cProfile.Profile()
        # c.enable()
        tstart = timer()
        fl = AllPairsShortestPath.floyd_warshall_simple_slow_parallel(graph, processes=4)
        fl_time = timer()-tstart
        # c.disable()
        # c.print_stats()

        tstart = timer()
        fl_np = AllPairsShortestPath.floyd_warshall_simple_slow(graph)
        fl_np_time = timer()-tstart

        print("parallel dij: ",fl_dij_time) 
        print("parallel fw: ",fl_time)
        print("sequential fw: ",fl_np_time)
        # compare results
        for x in fl_np:
            if round(fl_np[x]) != round(fl[x]) or round(fl_dij[x]) != round(fl_np[x]):
                print(x, fl_np[x], fl[x], fl_dij[x])
                raise ValueError

if __name__ == "__main__":
    unittest.main()

