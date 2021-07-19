import unittest
import os
import random 

from steinerpy.library.graphs.parser import DataParser
from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.context import Context

import steinerpy.config as cfg

cwd = cfg.pkg_dir
cfg.Algorithm.sstar_heuristic_type = "diagonal_nonuniform"
cfg.Animation.visualize = True

class TestParser(unittest.TestCase):

    # def test_simple_parser_steinlib(self):
    #     # define filename to use
    #     filename = os.path.join(cfg.data_dir,"steinlib/B/b01.stp")
    #     # Get package file location
    #     filepath = os.path.join(cwd, filename)
    #     # Return edge dictionary and terminal list
    #     g, T = DataParser.parse(filepath, dataset_type="steinlib")        
    #     # Create context, run and store results
    #     context = Context(g, T)
    #     context.run('Kruskal')
    #     results1 = context.return_solutions()
    #     context.run('S*-HS0')
    #     results2 = context.return_solutions()
    #     # assert results
    #     self.assertEqual(sum(results1['dist']), sum(results2['dist']))

    # def test_simple_parser_steinlib_format(self):
    #     # Some online_data options
        
    #     # define filename to use
    #     filename = os.path.join(cfg.data_dir,"simple/test.stp")
    #     # Get package file location
    #     filepath = os.path.join(cwd, filename)
    #     # Return edge dictionary and terminal list
    #     g, T = DataParser.parse(filepath, dataset_type="steinlib")        
    #     # Create context, run and store results
    #     context = Context(g, T)
    #     context.run('Kruskal')
    #     results1 = context.return_solutions()
    #     context.run('S*-HS0')
    #     results2 = context.return_solutions()
    #     # assert results
    #     self.assertEqual(sum(results1['dist']), sum(results2['dist'])) 

    def test_simple_parser_mapf(self):
        # Some online_data options
        # mapf

        # define filename to use
        # filename = "results/online_data/mapf/Berlin_1_256.map"
        # filename = "results/online_data/mapf/maze-32-32-2.map"
        filename = os.path.join(cfg.data_dir,"mapf", "maze-32-32-2.map")
        # Get package file location
        # cwd = os.path.dirname(os.path.realpath(steinerpy.__file__))+"/../"
        filepath = os.path.join(cwd, filename)

        # Return edge dictionary and terminal list
        # obs, height, width = DataParser.parse(filepath, dataset_type="mapf")
        graph = DataParser.parse(filepath, dataset_type="mapf")
        minX, maxX, minY, maxY = graph.grid_dim
        obs = graph.obstacles
        # # dims
        # minX, maxX = 0, width - 1
        # minY, maxY = 0, height - 1

        # grid_size = 1       # grid fineness[m]
        # grid_dim = [minX, maxX, minY, maxY]
        # n_type = 8           # neighbor type

        # # Create a squareGrid using GraphFactory
        # graph = GraphFactory.create_graph(type_="SquareGrid", grid_dim=grid_dim, grid_size=grid_size, n_type= n_type, obstacles=obs)      
        # graph.show_grid()

        # generate random terminals
        T = []
        while len(T) < 5:
            x = random.randint(minX, maxX)
            y = random.randint(minY, maxY)
            if (x,y) not in obs:
                T.append((x,y))

        # Create context, run and store results
        context = Context(graph, T)

        context.run('Kruskal')
        results1 = context.return_solutions()

        context.run('S*-HS')
        results2 = context.return_solutions()

        # assert results
        eps = 1e-6
        self.assertTrue( abs(sum(results1['dist']) - sum(results2['dist']) <= eps))

if __name__ == "__main__":
    unittest.main()