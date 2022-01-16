import unittest
import steinerpy.config as cfg
from steinerpy.algorithms import SstarMMLP, SstarMM

import os
from steinerpy.library.graphs.parser import DataParser

import random
random.seed(123)

# turn on visualizations for debugging
cfg.Animation.visualize=True

import matplotlib.pyplot as plt

class TestlbPropagation(unittest.TestCase):

    def setUp(self) -> None:
        # load map from disk
        map_file = os.path.join(cfg.data_dir, "mapf", "den312d.map")
        graph = DataParser.parse(map_file, dataset_type="mapf")
        self.graph = graph

        # get dim
        minX, maxX, minY, maxY = graph.grid_dim

        # generate random unique set of terminals
        T = set()
        while len(T)<15:
            x = random.randint(minX, maxX)
            y = random.randint(minY, maxY)
            if (x,y) not in graph.obstacles:
                T.add((x,y))

        # convert to list
        self.T = list(T)

    def test_mm_lb_prop_vs_normal(self):
        """Test meet-in-the-middle merged with lb propagation"""
        mm_lb = SstarMMLP(self.graph, self.T)
        mm_lb.run_algorithm()
        res = mm_lb.return_solutions()

        pass
        mm = SstarMM(self.graph, self.T)
        mm.run_algorithm()
        res = mm.return_solutions()

        plt.pause(30)

if __name__ == "__main__":
    unittest.main(verbosity=4)

