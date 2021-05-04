import unittest
from timeit import default_timer as timer

from steinerpy.library.search.search_algorithms import AStarSearch
from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.library.pipeline import OfflinePaths

# Create square grid using GraphFactory
minX = 0		# [m]
maxX = 7   
minY = 0
maxY = 7
grid = None         # pre-existing 2d numpy array?
grid_size = 1       # grid fineness[m]
grid_dim = [minX, maxX, minY, maxY]
n_type = 8           # neighbor type

# Create a squareGrid using GraphFactory
sq = GraphFactory.create_graph("SquareGrid", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type)      


class TestAllShortestPaths(unittest.TestCase):

    def test_slow_case(self):
        # Calculate all shortest paths
        startTime = timer()
        val = OfflinePaths.get_all_paths(sq)
        endTime = timer()
        print("test_slow_case time taken(s)", endTime - startTime)
        print(len(val))

    def test_fast_multiprocess_case(self):
        startTime = timer()
        val = OfflinePaths.get_all_paths_fast(sq)
        #end timer
        endTime = timer()
        print("test_fast_case time taken(s): ",endTime - startTime)
        print(len(val))

    # def test_multi_astar_run(self):

if __name__ == "__main__":
    unittest.main()