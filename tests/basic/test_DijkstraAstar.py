import unittest
from timeit import default_timer as timer

from steinerpy.library.search.search_algorithms import UniSearch
from steinerpy.library.graphs.graph import GraphFactory
import steinerpy.config as cfg

# Set delay as needed before importing AnimateV2
cfg.Animation.animate_delay = 0.0
from steinerpy.library.animation.animationV2 import AnimateV2

# Create a Square Graph
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

class TestDijkstraAstarSearch(unittest.TestCase):

    #run Dijkstra
    @unittest.skip("Demo purposes")
    def test_dijkstra_run(self):
        # cfg.animate_delay = 0.001

        start = (0,0)
        goal = (2,2)

        sTime = timer()
        search = UniSearch(sq, start, goal, "zero", visualize=True)
        parents, g = search.use_algorithm()
        eTime = timer()
        print("Dijkstra time taken(s):",eTime - sTime)
        AnimateV2.close()

    #run Astar
    @unittest.skip("Demo purposes")
    def test_astar_run(self):
        # cfg.animate_delay = 0.005
        start = (-5,-5)
        goal = (5,5)

        sTime = timer()
        search = UniSearch(sq, start, goal, "diagonal_nonuniform", visualize=True)
        parents, g = search.use_algorithm()
        eTime = timer()
        print("Dijkstra time taken(s):",eTime - sTime)
        AnimateV2.close()

if __name__=="__main__":
    unittest.main()

