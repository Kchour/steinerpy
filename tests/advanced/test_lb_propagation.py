import unittest
import steinerpy.config as cfg
from steinerpy.algorithms import SstarMMLP, SstarMM, SstarHSLP, SstarHS,\
                                SstarMMUNLP, SstarMMUN, Kruskal

import os
from steinerpy.library.graphs.parser import DataParser

import random
# random.seed(1)

# turn on visualizations for debugging
cfg.Animation.visualize = False

import matplotlib.pyplot as plt

import steinerpy as sp
l = sp.enable_logger()
sp.set_level(sp.WARN)

# add custom logging filter
import logging
class CustomFilter(logging.Filter):
    def filter(self, record):
        """Filter incoming messages
        record.name: name of string message usually the module name with dot notation
        record.message: string message
        
        """
        # if "MM CRITERIA, PATH_COST, RHS" in record.message:
        # if "Heuristic value, search_id" in record.message:
        if "Observing edge between" in record.message or "Adding sp edge between" in record.message \
            or "ITERATION" in record.message:
            return True
        else:
            return False

# l.handlers[1].addFilter(CustomFilter())


@unittest.skip("for now")
class TestlbPropagation(unittest.TestCase):

    def gen(self) -> None:
        # load map from disk
        map_file = os.path.join(cfg.data_dir, "mapf", "den312d.map")
        graph = DataParser.parse(map_file, dataset_type="mapf")
        self.graph = graph

        # get dim
        minX, maxX, minY, maxY = graph.grid_dim

        # generate random unique set of terminals
        T = set()
        while len(T)<10:
            x = random.randint(minX, maxX)
            y = random.randint(minY, maxY)
            if (x,y) not in graph.obstacles:
                T.add((x,y))

        # convert to list
        self.T = list(T)

    def tearDown(self) -> None:
        plt.close('all')

    @unittest.skip("not needed for now")
    def test_mm_lb_prop_vs_normal(self):
        """Test meet-in-the-middle merged with lb propagation"""
        mm_lb = SstarMMLP(self.graph, self.T)
        mm_lb.run_algorithm()
        res1 = mm_lb.return_solutions()

        pass
        mm = SstarMM(self.graph, self.T)
        mm.run_algorithm()
        res2 = mm.return_solutions()

        self.assertTrue(    abs(sum(res1['dist']) - sum(res2['dist'])) < 1e-6)

        
        print("MM--------------------------------\n") 
        print("with lb: ", res1['stats'])
        print("\n")
        print("normal: ", res2['stats'])
        if cfg.Animation.visualize:
            plt.pause(5)



    @unittest.skip("not needed for now")
    def test_hs_lb_prop_vs_normal(self):
        """Test POHL HS merged with lb propagation"""
        hs_lb = SstarHSLP(self.graph, self.T)
        hs_lb.run_algorithm()
        res1 = hs_lb.return_solutions()


        hs = SstarHS(self.graph, self.T)
        hs.run_algorithm()
        res2 = hs.return_solutions()

        self.assertTrue(    abs(sum(res1['dist']) - sum(res2['dist'])) < 1e-6)

        print("HS-------------------------------\n") 
        print("with lb: ", res1['stats'])
        print("\n")
        print("normal: ", res2['stats'])
        if cfg.Animation.visualize:
            plt.pause(5)

    @unittest.skip("not needed for now")
    def test_hs_weird_case(self):
        """Test POHL HS merged WEIRD CASE"""
        hs_lb = SstarHSLP(self.graph, [(24, 34), (22, 78)])
        hs_lb.run_algorithm()
        res1 = hs_lb.return_solutions()

        hs = SstarHSLP(self.graph, [(24, 34), (22, 78)])
        hs.run_algorithm()
        res2 = hs.return_solutions()

        self.assertTrue(    abs(sum(res1['dist']) - sum(res2['dist'])) < 1e-6)

        print("HS-------------------------------\n") 
        print("with lb: ", res1['stats'])
        print("\n")
        print("normal: ", res2['stats'])
        if cfg.Animation.visualize:
            plt.pause(5)

    @unittest.skip("not needed for now")
    def test_compare_mm_hs(self):
        """Test both mm and hs and lb variants"""
        mm_lb = SstarMMLP(self.graph, self.T)
        mm_lb.run_algorithm()
        res1 = mm_lb.return_solutions()

        mm = SstarMM(self.graph, self.T)
        mm.run_algorithm()
        res2 = mm.return_solutions()

        self.assertTrue(    abs(sum(res1['dist']) - sum(res2['dist'])) < 1e-6)

        """Test POHL HS merged with lb propagation"""
        hs_lb = SstarHSLP(self.graph, self.T)
        hs_lb.run_algorithm()
        res3 = hs_lb.return_solutions()

        self.assertTrue(    abs(sum(res2['dist']) - sum(res3['dist'])) < 1e-6)

        hs = SstarHS(self.graph, self.T)
        hs.run_algorithm()
        res4 = hs.return_solutions()

        self.assertTrue(    abs(sum(res3['dist']) - sum(res4['dist'])) < 1e-6)

        print("MM-------------------------------\n") 
        print("with lb: ", res1['stats'])
        print("\n")
        print("normal: ", res2['stats'])
        print("HS-------------------------------\n") 
        print("with lb: ", res3['stats'])
        print("\n")
        print("normal: ", res4['stats'])
        if cfg.Animation.visualize:
            plt.pause(5)

class TestUntilFailureDebug(unittest.TestCase):

    def load(self) -> None:
        # load map from disk
        map_file = os.path.join(cfg.data_dir, "mapf", "den312d.map")
        graph = DataParser.parse(map_file, dataset_type="mapf")
        self.graph = graph

        # get dim
        self.minX, self.maxX, self.minY, self.maxY = graph.grid_dim


    def gen(self):
        # generate random unique set of terminals
        T = set()
        # while len(T)<10:
        while len(T)<10:
            x = random.randint(self.minX, self.maxX)
            y = random.randint(self.minY, self.maxY)
            if (x,y) not in self.graph.obstacles:
                T.add((x,y))

        # convert to list
        self.T = list(T)

    # @unittest.skip("not working just yet")
    def test_until_failure_seed(self):
        # cnt = 21
        # cnt = 21    # un lb issue
        cnt = 70    # merged lb issue
        lim = 2000  
        # load map
        self.load()
        while cnt < lim:
            # keep track of seed
            random.seed(cnt)
            self.gen()
            print("current seed: ", cnt)
            self.with_unmerged_mm_and_lp()
            cnt +=1

    @unittest.skip("Resolved due to tolerance and inequality issue")
    def test_weird_case(self):
        """Test weird bounds edge case?"""
        self.load()
        T = [(54, 8),(53, 36)]

        # path cost should be 76.21
        # base = Kruskal(self.graph, T)
        # base.run_algorithm()
        # res0 = base.return_solutions()
        # print(sum(res0['dist']))

        mm = SstarMMUN(self.graph, T)
        mm.run_algorithm()
        res1 = mm.return_solutions()

        mm_lb = SstarMMUNLP(self.graph, T)
        mm_lb.run_algorithm()
        res2 = mm_lb.return_solutions()

        self.assertTrue(    abs(sum(res1['dist']) - sum(res2['dist'])) < 1e-6)

    def with_unmerged_mm_and_lp(self):
        """Test unmerged mm against lb prop""" 
        res1 = {'stats': None}
        res2 = {'stats': None}
        # base = Kruskal(self.graph, self.T)
        # base.run_algorithm()
        # res5 = base.return_solutions()

        print("MM UN lB: ".ljust(25), end=""),
        mm_lb = SstarMMUNLP(self.graph, self.T)
        mm_lb.run_algorithm()
        res1 = mm_lb.return_solutions()
        print(sum(res1['dist']), "".ljust(25), res1['stats']["expanded_nodes"])

        # self.assertTrue(    abs(sum(res5['dist']) - sum(res1['dist'])) < 1e-6)

        print("MM UN: ".ljust(25), end="")
        mm = SstarMMUN(self.graph, self.T)
        mm.run_algorithm()
        res2 = mm.return_solutions()
        print(sum(res2['dist']), "".ljust(25), res2['stats']["expanded_nodes"])

        self.assertTrue(    abs(sum(res1['dist']) - sum(res2['dist'])) < 1e-6)

        print("MM Merged LB: ".ljust(25), end="")
        mm_lb = SstarMMLP(self.graph, self.T)
        mm_lb.run_algorithm()
        res3 = mm_lb.return_solutions()
        print(sum(res3['dist']), "".ljust(25), res3['stats']["expanded_nodes"])

        # self.assertTrue(    abs(sum(res2['dist']) - sum(res3['dist'])) < 1e-6)

        print("MM Merged: ".ljust(25), end="")
        mm = SstarMM(self.graph, self.T)
        mm.run_algorithm()
        res4 = mm.return_solutions()
        print(sum(res4['dist']), "".ljust(25), res4['stats']["expanded_nodes"])

        self.assertTrue(    abs(sum(res3['dist']) - sum(res4['dist'])) < 1e-6)

        # print("Kruskal: ".ljust(25), end="")
        base = Kruskal(self.graph, self.T)
        base.run_algorithm()
        res5 = base.return_solutions()
        print("Kruskal: ", sum(res5['dist']))

        self.assertTrue(    abs(sum(res5['dist']) - sum(res4['dist'])) < 1e-6)

        print("MMUN--------------------------------\n") 
        print("with lb: ", res1['stats'])
        print("\n")
        print("normal: ", res2['stats'])
        print("MM--------------------------------\n") 
        print("with lb: ", res3['stats'])
        print("\n")
        print("normal: ", res4['stats'])
        if cfg.Animation.visualize:
            plt.pause(5)

if __name__ == "__main__":
    unittest.main(verbosity=2)

