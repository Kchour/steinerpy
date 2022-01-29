import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from steinerpy.environment import EnvType, EnvLoader 
from steinerpy.library.pipeline import GenerateResultsMulti, GenerateBaseLine, Process
from steinerpy.library.pipeline.r0generate_heuristics import GenerateHeuristics
import steinerpy.config as cfg

cfg.Animation.visualize = False 

import steinerpy as sp
sp.enable_logger()
sp.set_level(sp.WARN)

# for deterministic behavior
import random
random.seed(123)

# load heuristic preprocessed file
# with open("./heuristics/h_maze-32-32-4.map.pkl", 'rb') as f:
#     data = pickle.load(f)
# graph = EnvLoader.load(EnvType.MAPF, "maze-32-32-4.map")

with open("./heuristics/h_den520d.map.pkl", 'rb') as f:
    data = pickle.load(f)
graph = EnvLoader.load(EnvType.MAPF, "den520d.map")

# with open("./heuristics/h_Archipelago.map.pkl", 'rb') as f:
    # data = pickle.load(f)
# graph = EnvLoader.load(EnvType.GRID_2D, os.path.join("sc", "Archipelago.map"))

pass

# make sure you dont run the following if visualize=True
# graph.show_grid()
# for k in data.keys():
#     if k != "type":
#         plt.scatter(k[0], k[1])
# plt.show(block=False)
# pass

# use baseline to generate a random problem
gen_bs = GenerateBaseLine(graph=graph)
gen_bs.randomly_generate_instances(1, 10)
instances = gen_bs.instances

# try loading heuristics 
GenerateHeuristics.load_results(results=data)

# pass this pre run func to generator
def pre_run_func(self, *kwargs):
    """self refers to GenerateResultsMulti object"""
    # if "S*-BS" in self.alg or "S*-MM0" in self.alg:
    #     pass
    # elif "S*-MM2" in self.alg:
    #     self.alg = "S*-MM"
    #     # try changing heuristic type
    #     cfg.Algorithm.sstar_heuristic_type = "diagonal_nonuniform"
    #     pass
    # else:
    #     # make sure we compute bounds for cdh
    #     cfg.Algorithm.sstar_heuristic_type = "preprocess"
    #     GenerateHeuristics.cdh_compute_bounds(graph, self.terminals)
    
    # cfg.Algorithm.sstar_heuristic_type = "diagonal_nonuniform"
    cfg.Algorithm.sstar_heuristic_type = "preprocess"
    cfg.Algorithm.use_bpmx = True
    GenerateHeuristics.cdh_compute_bounds(graph, self.terminals)

# now pass instances to results generator
# algs = ["S*-MM", "S*-BS"]
# algs = ["S*-MM", 'S*-MM2']
# algs = ['S*-MM-LP']
algs = ['S*-MM']
# algs = ['S*-MM-UN']
gen_res = GenerateResultsMulti(graph=graph, algs_to_run=algs, pre_run_func=pre_run_func)
gen_res.input_specifed_instances(instances)
res = gen_res.run()

gen_proc = Process()
gen_proc.specify_data(main_results_data=res)
df = gen_proc.run()
print(df)
# np.array(list(data[(421, 17)]))
# process for printing


# graph.show_grid()
# for v in data.values():
#     if v != "CDH":
#         test = np.array(list(v))
#         plt.scatter(test[:,0], test[:,1])
# plt.show(block=False)
# pass 

# total number of keys = 110

# # tranpose method (to all states vs. landmarks: dist)
# new_data = {}
# for pivot, values in data.items():
#     if pivot == "type":
#         new_data[pivot] = values
#         continue

#     for state, dist in values.items():

#         if state not in new_data:
#             new_data[state] = {pivot: dist}
#         else:
#             new_data[state].update({pivot: dist})

# inverse transpose (landmarks vs. states: dist)
