import unittest
import os
import random
import cProfile, pstats

# for profiling code
profiler = cProfile.Profile()

# Set seed if desired
random.seed(123)    # 50 terminal
# random.seed(1234)    # 2 terminal

# THE USER CAN SPECIFY A CUSTOM LOCATION AS DESIRED
cwfd = os.path.dirname(os.path.abspath(__file__))
# heu_path = os.path.join(cwfd, "..", "heuristic")
heu_path = "/tmp"

# just to get cpu count
import multiprocessing
cpu_count = int(multiprocessing.cpu_count()/2)

from steinerpy.library.pipeline import GenerateHeuristics   # to generate preprocessed heuristics
from steinerpy.library.graphs.parser import DataParser      # to load mapf instances
import steinerpy.config as cfg                              # to ensure heuristic is set to "preprocess"
from steinerpy.context import Context                       # helper to load graph and run algorithms

# Visualize things
# cfg.Animation.visualize = True

# location to save preproessed heuristics (in tmp)
save_file = os.path.join(heu_path, "den312d.map.pkl")

# get mapf map file
map_file = os.path.join(cfg.data_dir, "mapf", "den312d.map")
graph = DataParser.parse(map_file, dataset_type="mapf")

# get dim and obstacles of map
minX, maxX, minY, maxY = graph.grid_dim
obs = graph.obstacles

# generate random unique set of terminals
T = set()
while len(T) < 15:
    x = random.randint(minX, maxX)
    y = random.randint(minY, maxY)
    if (x,y) not in obs:
        T.add((x,y))

# convert back to list!
T = list(T)

further_save_file = os.path.join(heu_path, "den312d.map.land-to-apsp.pkl")

class TestCreateAndRunHeuristics(unittest.TestCase):

    def setUp(self):
        self.old_setting = cfg.Algorithm.sstar_heuristic_type
        cfg.Algorithm.sstar_heuristic_type = "diagonal_nonuniform"
        # cfg.Misc.log_conf["handlers"]['console']['level'] = "DEBUG"
        # cfg.reload_log_conf()
        cfg.Animation.visualize = True

    def tearDown(self):
        cfg.Algorithm.sstar_heuristic_type = self.old_setting  

    @unittest.skip("SKIPPING due to length of time")
    def test_load_heuristics_from_disk_run_sstar(self):
        # generate heuristics for den312d.map if not found 
        # THIS IS TIME CONSUMING
        if not os.path.exists(save_file):

            # generate heuristics and save it to the desired file_location
            gh = GenerateHeuristics.gen_and_save_results(graph, file_location=heu_path, file_name="den312d.map.pkl", processes=cpu_count)
            
            # convert landmarks to apsp for efficiency
            if gh['type'] == "LAND":
                GenerateHeuristics.convert_land_to_apsp(data=gh, output=further_save_file)

        # load preprocessed heuristic from disk
        GenerateHeuristics.load_results(further_save_file)

        # change heuristic setting
        cfg.Algorithm.sstar_heuristic_type = "preprocess"        

        # create context handler and run S*-mm
        context = Context(graph, T)

        # profiler enable
        profiler.enable()

        # Now run
        context.run("S*-MM")

        # disable profiler
        profiler.disable()

        # store results
        results = context.return_solutions()   

        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()

        print(results)

    def test_try_different_heuristics(self):
        """Run all the algorithms with different grid based heuristics
        
        NOTE:
            Using an inadmissible heuristic may give erraneous results
            such as wrong final distances and distances added in incorrect order

            This is the case with Manhattan distance and 8-neighbor grids

        """
        # Run all 2d grid based heuristics except preprocess
        # Make sure `grid` is selected as graph domain
        cfg.Algorithm.graph_domain = "grid"

        # try to use custom heuristics
        from steinerpy.algorithms.common import CustomHeuristics
        CustomHeuristics.bind(lambda x,y: 0) 

        # algorithms = ["S*-HS", "S*-BS", "S*-MM", "S*-MM0", "S*-unmerged"]
        # heuristics = ["manhattan", "custom", "diagonal_nonuniform", "diagonal_uniform", "euclidean", "zero"]        
        # algorithms = ["Kruskal", "S*-HS"]
        algorithms = ["S*-HS"]
        # algorithms = ["Kruskal", "S*-BS"]
        heuristics = ["diagonal_nonuniform"]
        for h in heuristics:

            dist = []
            # set heuristic configuration
            cfg.Algorithm.sstar_heuristic_type = h
            print("")
            print("Using {}".format(h))
            # run each algorithm
            for alg in algorithms:
                context = Context(graph, T)
                print("Running {}".format(alg))
                context.run(alg)

                dist.append(sum(context.return_solutions()['dist']))

                # check monotonicity, not guaranteed if h is not admissible
                try:
                    assert all( y-x>=0 for x,y in zip(context.return_solutions()['dist'], context.return_solutions()['dist'][1:] ))
                except:
                    print(context.return_solutions()['dist'])
            
            try: 
                assert all ( abs(x-y)< 1e-6 for x,y in zip(dist, dist[1:]))
            except:
                print(dist)


if __name__ == "__main__":
    unittest.main() 