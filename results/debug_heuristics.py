import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from steinerpy.environment import EnvType, EnvLoader 
from steinerpy.library.pipeline import GenerateResultsMulti, GenerateBaseLine, Process
from steinerpy.library.pipeline.r0generate_heuristics import GenerateHeuristics
import steinerpy.config as cfg

# visualize algorithm?
cfg.Animation.visualize = False
# visualize bounds function?
cfg.Pipeline.debug_vis_bounds = False
# profile the code
cfg.Misc.profile_frame = False
# reprioritize after merge?
cfg.Algorithm.reprioritize_after_merge = True

import steinerpy as sp
sp.enable_logger()
sp.set_level(sp.WARN)

# for deterministic behavior
import random
# random.seed(123)
random.seed(1)
# rng = np.random.default_rng(seed=1)
# out = rng.random(5)
np.random.seed(1)

# load heuristic preprocessed file
# with open("./heuristics/h_maze-32-32-4.map.pkl", 'rb') as f:
#     data = pickle.load(f)
# graph = EnvLoader.load(EnvType.MAPF, "maze-32-32-4.map")

# with open("./heuristics/h_den520d.map.pkl", 'rb') as f:
#     data = pickle.load(f)
# graph = EnvLoader.load(EnvType.MAPF, "den520d.map")

# with open("./heuristics/h_Archipelago.map.pkl", 'rb') as f:
#     data = pickle.load(f)
# graph = EnvLoader.load(EnvType.GRID_2D, os.path.join("sc", "Archipelago.map"))

with open("./heuristics/h_Complex.3dmap.pkl", 'rb') as f:
    data = pickle.load(f)
graph = EnvLoader.load(EnvType.GRID_3D, "Complex.3dmap")
pass

# with open("./heuristics/h_Simple.3dmap.pkl", 'rb') as f:
#     data = pickle.load(f)
# graph = EnvLoader.load(EnvType.GRID_3D, "Simple.3dmap")
# pass

# viewing graph
# graph.show_grid()

cfg.Pipeline.min_reach_pivots = 4

# # make sure you dont run the following if visualize=True
# graph.show_grid()
# for k in data.keys():
#     # plot all surrogate states
#     if k != "type":
#         plt.scatter(k[0], k[1])
# plt.show(block=False)
# pass

# use baseline to generate a random problem
gen_bs = GenerateBaseLine(graph=graph)
gen_bs.randomly_generate_instances(1, 20)
instances = gen_bs.instances

# very specific instance
# instances = [(37, 72, 9), (16, 129, 76)]
# instances = [[(37, 72, 9), (16,129,76)]]
# instances = [[(84, 50, 68), (20, 101, 18), (37, 72, 9), (16, 129, 76)]]
# instances = [[(84,50,68), (20, 101, 18), (37, 72, 9),]] 

# try loading heuristics 
GenerateHeuristics.load_results(results=data)
# specify heuristic database type

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
    
    # for grid_2d/mapf instances only!
    # cfg.Algorithm.sstar_heuristic_type = "diagonal_nonuniform"

    # for grid_3d only
    # cfg.Algorithm.sstar_heuristic_type = "voxel"

    GenerateHeuristics.preload_type="CDH"
    cfg.Algorithm.sstar_heuristic_type = "preprocess"
    cfg.Algorithm.use_bpmx = True
    GenerateHeuristics.cdh_compute_bounds(graph, self.terminals)

# now pass instances to results generator
# algs = ["S*-MM", "S*-BS"]
# algs = ["S*-MM", 'S*-MM2']
# algs = ['S*-MM-LP']
# algs = ['S*-HS']
# algs = ['S*-MM-UN']
algs = ['S*-MM']
gen_res = GenerateResultsMulti(graph=graph, algs_to_run=algs, pre_run_func=pre_run_func)
gen_res.input_specifed_instances(instances)
res = gen_res.run()

gen_proc = Process()
gen_proc.specify_data(main_results_data=res)
cost_df, time_df = gen_proc.run()
print(cost_df)
print(time_df)
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
