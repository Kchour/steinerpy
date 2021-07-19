import unittest
import os
import random
import cProfile, pstats

# for profiling code
profiler = cProfile.Profile()

# Set seed if desired
random.seed(123)    # 50 terminal
# random.seed(1234)    # 2 terminal


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
while len(T) < 50:
    x = random.randint(minX, maxX)
    y = random.randint(minY, maxY)
    if (x,y) not in obs:
        T.add((x,y))

# convert back to list!
T = list(T)

further_save_file = os.path.join(heu_path, "den312d.map.land-to-apsp.pkl")

# generate heuristics for den312d.map if not found 
if not os.path.exists(save_file):

    # generate heuristics and save it to the desired file_location
    gh = GenerateHeuristics.gen_and_save_results(graph, file_location=heu_path, file_name="den312d.map.pkl", processes=cpu_count)
    
    # convert landmarks to apsp for efficiency
    if gh['type'] == "LAND":
        GenerateHeuristics.convert_land_to_apsp(data=gh, output=further_save_file)


class TestGenerateHeuristics(unittest.TestCase):

    def test_load_heuristics_from_disk_run_sstar(self):

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

if __name__ == "__main__":
    unittest.main() 