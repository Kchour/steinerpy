import unittest
from unittest.mock import Mock, patch
# Some examples https://gist.github.com/vkroz/a59c7e05014e456f86e0

from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.library.search.search_algorithms import MultiSearch

class TestGraph(unittest.TestCase):

    @unittest.skip("temp")
    def test_create_square_grid(self):
        ''' Test GraphFactory in creating a SquareGrid object '''
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
        sq = GraphFactory.create_graph("SquareGrid", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type)      

        # test to see if sq is an instance of SquareGrid
        from steinerpy.library.graphs.graph import SquareGrid
        self.assertTrue(isinstance(sq, SquareGrid))

        # test if instance values are correct
        self.assertEqual(sq.grid_size, grid_size)
        self.assertEqual(sq.grid_dim, grid_dim)
        self.assertEqual(sq.neighbor_type, n_type)

        # Define obstacles (physical coordinates, origin is lower-left)
        obstacles = [(x, 0) for x in range(-10,10,1)]
        obstacles.extend([(0, y) for y in range(-10,10,1)])

        # We can either add obstacles using 'set_obstacle' method, or do it with GraphFactory
        sq.set_obstacles(obstacles)

        # Show image (comment this out if it blocks)
        sq.show_grid()
        # print("")

         # get all graphs nodes
        nodes = sq.get_nodes()
        # print("")

        # get all edges
        edges = sq.get_edges()
        # print(sq.edge_count(), len(edges))

        # adjacency matrix
        adj = sq.get_adjacency_matrix()
        # print("")

        # TODO: test the other functions in SquareGrid

    @unittest.skip("testing")
    def test_create_generic(self):
        ''' test GraphFactory in creating a Generic MyGraph object '''
        from steinerpy.library.graphs.graph import MyGraph

        # Define some edges
        edgeDict = {('v1','v2'): 1,
                    ('v2','v3'): 1,
                    ('v3','v4'): 1,
                    ('v4','v5'): 1,
                    ('v5','v6'): 1,
                    ('v6','v7'): 1,
                    ('v7','v8'): 1,
                    ('v8','v5'): 1}
        
        # Create a generic graph using factory method
        genG = GraphFactory.create_graph("Generic", edge_dict = edgeDict, graph_type = "undirected", visualize=False)

        # test to see if genG is an instance of MyGraph
        self.assertTrue(isinstance(genG, MyGraph))

        # TODO return some neighbors and stuff

    def test_generic_search_no_h_dijkstra(self):
        # create a generic graph

        edgeDict = {('A', 'B'): 3,
                    ('A', 'C'): 1,
                    ('B', 'C'): 7,
                    ('B', 'D'): 5,
                    ('B', 'E'): 1,
                    ('C', 'D'): 2,
                    ('D', 'E'): 7
        }

        genG = GraphFactory.create_graph("Generic", edge_dict = edgeDict, graph_type="undirected", visualize=False)

        gs = MultiSearch(genG, 'C')

        # run updates 
        while not gs.frontier.empty():
            gs.update()

        self.assertEqual(gs.g['C'], 0)
        self.assertEqual(gs.g['A'], 1)
        self.assertEqual(gs.g['B'], 4)
        self.assertEqual(gs.g['D'], 2)
        self.assertEqual(gs.g['E'], 5)

if __name__ == "__main__":
    unittest.main()

