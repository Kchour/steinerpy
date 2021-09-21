import unittest
import os
import sys
import random
import pickle
import logging

import steinerpy.config as cfg
# cfg.Animation.visualize = True
# cfg.Misc.profile_frame = True

from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.library.graphs.parser import DataParser
from steinerpy.library.pipeline import GenerateResultsMulti, GenerateBaseLine, Process
from steinerpy.algorithms import SstarHS, SstarBS, SstarMM, SstarMM0, Kruskal

my_logger = logging.getLogger(__name__)
# # optionally use context to choose algorithms
# from steinerpy import context

# set seed if desired
random.seed(12)

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
# test_maps_mapf = [sq, mapf_maze, mapf_den]
test_maps_mapf = [sq, mapf_maze, mapf_den]

# load steinlib graphs
test_maps_steinlib = []
test_terminals_steinlib = []
stein_dir = {}
for root, dirs, files in os.walk(os.path.join(cfg.data_dir, "steinlib", "B")):
    for fname in files:
        # ensure proper file extension
        if "stp" in fname:
            sl_g, sl_terminals = DataParser.parse(os.path.join(root, fname), dataset_type="steinlib")
            sl_g.name = fname

            # store 
            test_maps_steinlib.append(sl_g)
            test_terminals_steinlib.append(sl_terminals)
            stein_dir[fname] = {'dir': os.path.join(root, fname), 'map': sl_g, 'terminals': sl_terminals}

# location to save cache files! WARNING: generating cache can take a long time
temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")

# get current working directory (the test folder)
cwd = os.path.dirname(os.path.abspath(__file__))

class TestGenerateAndCompareResultsMAPFGridBase(unittest.TestCase):
    """Randomly generate terminals and see if the answers will match!
        You can have the option of skipping baseline generation
        if the "XXX_baseline.pkl" file has already been generated
        for a particular instance-term combination
    """

    def setUp(self):
        self.old_setting = cfg.Algorithm.sstar_heuristic_type
        cfg.Algorithm.sstar_heuristic_type = "diagonal_nonuniform"
        # cfg.Misc.log_conf["handlers"]['console']['level'] = "DEBUG"
        # cfg.reload_log_conf()
        # cfg.Animation.visualize = True

    def tearDown(self):
        cfg.Algorithm.sstar_heuristic_type = self.old_setting  

    @unittest.skip("not testing right now")
    def test_generate_randomized_terminals_results_compare_mapf(self):
        # This heuristic is good for 8-neighbor square grids
        cfg.Algorithm.sstar_heuristic_type = "diagonal_nonuniform"

        # try reprioritzing
        # cfg.Algorithm.reprioritize_after_sp = False       #default
        # cfg.Algorithm.reprioritize_after_merge = True       #default

        num_of_inst = 1
        num_of_terms = 50
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
            algs_to_run = ["S*-BS", "S*-HS", "S*-MM", "S*-MM0", "S*-BS-UN", "S*-HS-UN", "S*-MM-UN", "S*-MM0-UN"]


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

                # test for monotonicity
                try:
                    assert all( y-x>=0 for x,y in zip(main_results['solution'][alg][0]['dist'],main_results['solution'][alg][0]['dist'][1:] ))
                except:
                    print("alg {} dist {}".format(alg, main_results['solution'][alg][0]['dist']))
                    print("kruskal order: {}".format(kruskal_results['solution'][0]['dist'] ))

            # process them
            save_path = os.path.join(cwd, "".join((_map.name, "processed_rand_results_test.xlsx")))
            pr = Process(save_path, file_behavior="OVERWRITE")
            pr.specify_files(baseline_save_path, main_save_path)
            pr.run()


class TestGenerateRandomResultsSteinLibGenericGraph(unittest.TestCase):

    def setUp(self):
        self.old_setting = cfg.Algorithm.sstar_heuristic_type
        from steinerpy.heuristics import Heuristics
        cfg.Algorithm.graph_domain = "generic"
        
        self.old_setting_domain = cfg.Algorithm.graph_domain 

        cfg.Misc.log_conf["handlers"]['console']['level'] = "WARN"
        cfg.reload_log_conf()
        # cfg.Animation.visualize = True
        Heuristics.bind(lambda next, goal: 0)

    def tearDown(self):
        cfg.Algorithm.sstar_heuristic_type = self.old_setting  
        cfg.Algorithm.graph_domain = self.old_setting_domain

    @unittest.skip("some issues yet")
    def test_generate_randomized_terminals_results_compare_steinlib(self):
        # from steinerpy.algorithms.common import CustomHeuristics
        # cfg.Algorithm.graph_domain = "generic"
        # zero_h = lambda *x, **kwargs: 0
        # CustomHeuristics.bind(zero_h)

        for _map, _terminals in zip(test_maps_steinlib, test_terminals_steinlib):
            baseline_save_path = os.path.join(cwd, "".join((_map.name, "_baseline.pkl")))
            gen_bs = GenerateBaseLine(graph=_map, save_path=baseline_save_path, file_behavior="SKIP", load_from_disk=True)
            # Get terminals
            gen_bs.input_specifed_instances([_terminals])
            # run the generator
            kruskal_results = gen_bs.run()
            # save instances
            instances = gen_bs.instances    

            # Generate results
            main_save_path = os.path.join(cwd, "".join((_map.name, "_main_results.pkl")))
            algs_to_run = ["S*-BS", "S*-HS", "S*-MM", "S*-MM0", "S*-BS-UN", "S*-HS-UN", "S*-MM-UN", "S*-MM0-UN"]
            # algs_to_run = ["S*-BS-UN", "S*-MM-UN", "S*-MM0-UN", "S*-HS-UN"]
            # algs_to_run = ["S*-HS"]
            gen_rm = GenerateResultsMulti(graph=_map, save_path=main_save_path, file_behavior="OVERWRITE", algs_to_run=algs_to_run)
            # specify instances
            gen_rm.input_specifed_instances([_terminals])
            # run the generator
            main_results = gen_rm.run()

            # loop over instances and compare mst values
            # store mst values for instance
            kruskal_value = sum(kruskal_results['solution'][0]['dist'])

            # loop over algorithms
            issue = False
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

                # test for monotonicity
                try:
                    assert all( y-x>=0 for x,y in zip(main_results['solution'][alg][0]['dist'],main_results['solution'][alg][0]['dist'][1:] ))
                except:
                    print("alg {} dist {}".format(alg, main_results['solution'][alg][0]['dist']))
                    issue = True
            
            # print kruskal order for answer
            if issue:
                print("kruskal order: {}".format(kruskal_results['solution'][0]['dist'] ))
             
            # process them
            save_path = os.path.join(cwd, "".join((_map.name, "processed_rand_results_test.xlsx")))
            pr = Process(save_path, file_behavior="OVERWRITE")
            pr.specify_files(baseline_save_path, main_save_path)
            pr.run()

    @unittest.skip("Not testing")
    def test_fixture_debug_issues(self):
        # b15.stp, b18.stp
        # stein_dir[fname] = {'dir': os.path.join(root, fname), 'map': sl_g, 'terminals': sl_terminals}
        map_name = "b02.stp"
        gen_bs = GenerateBaseLine(graph=stein_dir[map_name]['map'])
        # get terminals
        gen_bs.input_specifed_instances([stein_dir[map_name]['terminals']])
        # run generator
        kruskal_results = gen_bs.run()
        
        # generator results
        algs_to_run = ["S*-BS", "S*-HS", "S*-MM", "S*-MM0", "S*-BS-UN", "S*-HS-UN", "S*-MM-UN", "S*-MM0-UN"]
        # algs_to_run = ["S*-BS", "S*-HS-UN"]
        # algs_to_run = ["S*-HS-UN"]
        gen_mr = GenerateResultsMulti(graph=stein_dir[map_name]["map"], algs_to_run=algs_to_run)
        gen_mr.input_specifed_instances([stein_dir[map_name]["terminals"]])
        main_results = gen_mr.run()

        # loop over instances and compare mst values
        # store mst values for instance
        kruskal_value = sum(kruskal_results['solution'][0]['dist'])

        # loop over algorithms
        issue = False
        for alg in algs_to_run:
            print("running {}".format(alg))
            alg_value = sum(main_results['solution'][alg][0]['dist'])
            # now compare all values in mst_values
            try:
                if abs(kruskal_value-alg_value) > 1e-6:
                    print (alg, main_results['terminals'][0], kruskal_value, alg_value)
                    raise ValueError("MST VALUES DONT MATCH")
            except:
                my_logger.error("much badness during 'test_generate_randomized_terminals_results_compare_steinlib'", exc_info=True)
                raise 

            # test for monotonicity
            try:
                assert all( y-x>=0 for x,y in zip(main_results['solution'][alg][0]['dist'],main_results['solution'][alg][0]['dist'][1:] ))
            except:
                print("alg {} dist {}".format(alg, main_results['solution'][alg][0]['dist']))
                issue = True
        
            # print kruskal order for answer
            if issue:
                print("Kruskal")
                for e, cost in zip(kruskal_results['solution'][0]['sol'] , kruskal_results['solution'][0]['dist']):
                    print(e, cost)

                print("main alg")
                for e, cost in zip(main_results['solution'][alg][0]['sol'] , main_results['solution'][alg][0]['dist']):
                    print(e, cost)

                raise


if __name__ == "__main__":
    unittest.main()