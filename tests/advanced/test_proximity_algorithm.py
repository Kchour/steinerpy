from logging import log
from steinerpy.library.misc.utils import Progress
import unittest

import steinerpy.config as cfg


from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.algorithms.proximity import Proximity, Unmerged

class TestSteinerUnmerged(unittest.TestCase):
    
    def setUp(self) -> None:
        
        self.old_vis = cfg.Animation.visualize 
        self.old_h = cfg.Algorithm.sstar_heuristic_type
        self.old_log_conf = cfg.Misc.log_conf.copy()

        cfg.Animation.visualize = True
        cfg.Algorithm.sstar_heuristic_type = "zero"
         
        cfg.Misc.log_conf["handlers"]["console"]["level"] = "DEBUG"
        cfg.reload_log_conf()

    def tearDown(self) -> None:
        cfg.Animation.visualize = self.old_vis
        cfg.Algorithm.sstar_heuristic_type = self.old_h
        cfg.Misc.log_conf = self.old_log_conf
        cfg.reload_log_conf()

    def test_proximity_algorithm_(self):

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
  
          # Create Astar object
        ao = Proximity(graph, terminals)

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