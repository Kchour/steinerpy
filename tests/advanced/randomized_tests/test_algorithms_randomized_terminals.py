import unittest
import os
import sys
import random
import pickle
import logging

import steinerpy.config as cfg
# cfg.Animation.visualize = False

from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.library.graphs.parser import DataParser
from steinerpy.library.pipeline import GenerateResultsMulti, GenerateBaseLine, Process
from steinerpy.algorithms import SstarHS, SstarBS, SstarMM, SstarMM0, Kruskal

my_logger = logging.getLogger(__name__)
# # optionally use context to choose algorithms
# from steinerpy import context

# set seed if desired
random.seed(123)

# Create square grid using GraphFactory
minX = -25			# [m]
maxX = 25   
minY = -25
maxY = 25
grid = None         # pre-existing 2d numpy array?
grid_size = 1       # grid fineness[m]
grid_dim = [minX, maxX, minY, maxY]
n_type = 8           # neighbor type

# Create a squareGrid using GraphFactory
sq = GraphFactory.create_graph("SquareGrid", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type)      
sq.name = "square"

# Load other maps: MAPF

mapf_maze = DataParser.parse(os.path.join(cfg.data_dir,"mapf", "maze-32-32-2.map"), dataset_type="mapf")
mapf_maze.name = "maze"
mapf_den = DataParser.parse(os.path.join(cfg.data_dir,"mapf", "den312d.map"), dataset_type="mapf")
mapf_den.name = "den"
# store maps
test_maps_mapf = [mapf_maze, mapf_den]

# load steinlib graphs
test_maps_steinlib = []
test_terminals_steinlib = []
for root, dirs, files in os.walk(os.path.join(cfg.data_dir, "steinlib", "B")):
    for fname in files:
        # ensure proper file extension
        if "stp" in fname:
            sl_g, sl_terminals = DataParser.parse(os.path.join(root, fname), dataset_type="steinlib")
            sl_g.name = fname

            # store 
            test_maps_steinlib.append(sl_g)
            test_terminals_steinlib.append(sl_terminals)

# location to save cache files! WARNING: generating cache can take a long time
temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")

# get current working directory (the test folder)
cwd = os.path.dirname(os.path.abspath(__file__))

class TestGenerateAndCompareResults(unittest.TestCase):
    """Randomly generate terminals and see if the answers will match!
        You can have the option of skipping baseline generation
        if the "XXX_baseline.pkl" file has already been generated
        for a particular instance-term combination
    """
    # @unittest.skip("TEST")
    def test_generate_randomized_terminals_results_compare_mapf(self):
        cfg.Algorithm.sstar_heuristic_type = "diagonal_nonuniform"

        num_of_inst = 25
        num_of_terms = 12
        for _map in test_maps_mapf:
            baseline_save_path = os.path.join(cwd, "".join((_map.name, "_baseline.pkl")))
            gen_bs = GenerateBaseLine(graph=_map, save_path=baseline_save_path, file_behavior="OVERWRITE", load_from_disk=True)
            # generate random instances
            gen_bs.randomly_generate_instances(num_of_inst, num_of_terms)
            # run the generator
            kruskal_results = gen_bs.run()
            # save instances
            instances = gen_bs.instances    

            # Generate results
            main_save_path = os.path.join(cwd, "".join((_map.name, "_main_results.pkl")))
            algs_to_run = ["S*-BS", "S*-HS", "S*-MM", "S*-MM0"]
            gen_rm = GenerateResultsMulti(graph=_map, save_path=main_save_path, file_behavior="OVERWRITE", algs_to_run=algs_to_run)
            # specify instances
            gen_rm.input_specifed_instances(instances)
            # run the generator
            main_results = gen_rm.run()

            # loop over instances and compare mst values
            for ndx in range(len(instances)):
                # store mst values for instance
                kruskal_value = sum(kruskal_results['solution'][ndx]['dist'])

                # loop over algorithms
                for alg in algs_to_run:
                    alg_value = sum(main_results['solution'][alg][ndx]['dist'])
                    # now compare all values in mst_values
                    try:
                        if abs(kruskal_value-alg_value) > 1e-6:
                            print (alg, main_results['terminals'][ndx], kruskal_value, alg_value)
                            raise ValueError("MST VALUES DONT MATCH")
                    except:
                        my_logger.error("much badness during 'test_generate_randomized_terminals_results_compare_mapf'", exc_info=True)
                        raise 
             
            # process them
            save_path = os.path.join(cwd, "".join((_map.name, "processed_rand_results_test.xlsx")))
            pr = Process(save_path, file_behavior="OVERWRITE")
            pr.specify_files(baseline_save_path, main_save_path)
            pr.run()
    
    def test_generate_randomized_terminals_results_compare_steinlib(self):
        from steinerpy.algorithms.common import Common
        cfg.Algorithm.graph_domain = "generic"
        zero_h = lambda *x, **kwargs: 0
        Common.custom_heuristics = zero_h

        for _map, _terminals in zip(test_maps_steinlib, test_terminals_steinlib):
            baseline_save_path = os.path.join(cwd, "".join((_map.name, "_baseline.pkl")))
            gen_bs = GenerateBaseLine(graph=_map, save_path=baseline_save_path, file_behavior="SKIP", load_from_disk=True)
            # generate random instances
            gen_bs.input_specifed_instances([_terminals])
            # run the generator
            kruskal_results = gen_bs.run()
            # save instances
            instances = gen_bs.instances    

            # Generate results
            main_save_path = os.path.join(cwd, "".join((_map.name, "_main_results.pkl")))
            algs_to_run = ["S*-BS", "S*-HS", "S*-MM", "S*-MM0"]
            gen_rm = GenerateResultsMulti(graph=_map, save_path=main_save_path, file_behavior="OVERWRITE", algs_to_run=algs_to_run)
            # specify instances
            gen_rm.input_specifed_instances([_terminals])
            # run the generator
            main_results = gen_rm.run()

            # loop over instances and compare mst values
            # store mst values for instance
            kruskal_value = sum(kruskal_results['solution'][0]['dist'])

            # loop over algorithms
            for alg in algs_to_run:
                alg_value = sum(main_results['solution'][alg][0]['dist'])
                # now compare all values in mst_values
                try:
                    if abs(kruskal_value-alg_value) > 1e-6:
                        print (alg, main_results['terminals'][0], kruskal_value, alg_value)
                        raise ValueError("MST VALUES DONT MATCH")
                except:
                    my_logger.error("much badness during 'test_generate_randomized_terminals_results_compare_steinlib'", exc_info=True)
                    raise 
             
            # process them
            save_path = os.path.join(cwd, "".join((_map.name, "processed_rand_results_test.xlsx")))
            pr = Process(save_path, file_behavior="OVERWRITE")
            pr.specify_files(baseline_save_path, main_save_path)
            pr.run()



if __name__ == "__main__":
    unittest.main()