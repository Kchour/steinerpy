import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from steinerpy.environment import EnvType, EnvLoader 
from steinerpy.library.pipeline import GenerateResultsMulti, GenerateBaseLine, Process
from steinerpy.library.pipeline.r0generate_heuristics import GenerateHeuristics
import steinerpy.config as cfg

# required for 3d graph
from mayavi import mlab

# visualize algorithm?
cfg.Animation.visualize = False
# visualize bounds function?
cfg.Pipeline.debug_vis_bounds = False
# profile the code
cfg.Misc.profile_frame = False
# reprioritize after merge?
cfg.Algorithm.reprioritize_after_merge = True
# reprioritize after finding shortest paths
cfg.Algorithm.reprioritize_after_sp = False

import steinerpy as sp
sp.enable_logger()
sp.set_level(sp.WARN)

# for deterministic behavior
import random
# random.seed(123)
random.seed(69)
# rng = np.random.default_rng(seed=1)
# out = rng.random(5)
np.random.seed(69)

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

# cfg.Pipeline.min_reach_pivots = 10

with open("./heuristics/h_Complex.3dmap.pkl", 'rb') as f:
    data = pickle.load(f)
graph = EnvLoader.load(EnvType.GRID_3D, "Complex.3dmap")
pass

# try visualizing graph and pivots
xx, yy, zz = np.where(graph.grid>0)
mlab.points3d(xx, yy, zz, mode="cube", color=(0.75,0.75,0.75))
# try visualizing surrogates
blah = set()
for k, values in data["table"].items():
    # if k == "type":
    #     continue
    for v in values:
        blah.add(k)

# try visualizing pivots
piv = np.array(list(data["pivot_index"]))
surr = np.asarray(list(blah), dtype=np.int64)
mlab.points3d(piv[:,0], piv[:,1], piv[:,2], mode="sphere", color=(0,0,1), scale_factor=2)
mlab.points3d(surr[:,0], surr[:,1], surr[:,2], mode="sphere", color=(1,0,0), scale_factor=1)
mlab.show()


# with open("./heuristics/h_Simple.3dmap.pkl", 'rb') as f:
#     data = pickle.load(f)
# graph = EnvLoader.load(EnvType.GRID_3D, "Simple.3dmap")
# pass

# with open("./heuristics/h_FA1.3dmap.pkl", 'rb') as f:
#     data = pickle.load(f)
# graph = EnvLoader.load(EnvType.GRID_3D, "FA1.3dmap")
# pass

# viewing graph
# graph.show_grid()

cfg.Pipeline.min_reach_pivots = 16

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
gen_bs.randomly_generate_instances(1, 10)
res_bs = gen_bs.run()
instances = gen_bs.instances

# very specific instance
# instances = [[(0, 0, 0), (245,153,204)]]
# instances = [(37, 72, 9), (16, 129, 76)]
# instances = [[(37, 72, 9), (16,129,76)]]
# instances = [[(84, 50, 68), (20, 101, 18), (37, 72, 9), (16, 129, 76)]]
# instances = [[(84,50,68), (20, 101, 18), (37, 72, 9),]] 

# try loading heuristics 
GenerateHeuristics.load_results(results=data)
# specify heuristic database type

# pass this pre run func to generator
def pre_run_func(graph, terminals):
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
    cfg.Algorithm.sstar_heuristic_type = "voxel"

    # GenerateHeuristics.preload_type="CDH"
    # cfg.Algorithm.sstar_heuristic_type = "preprocess"
    # cfg.Algorithm.use_bpmx = True
    # GenerateHeuristics.cdh_compute_bounds(graph, terminals)

# now pass instances to results generator
# algs = ["S*-MM", "S*-BS"]
# algs = ["S*-MM", 'S*-MM2']
# algs = ['S*-MM-LP']
# algs = ['S*-HS']
# algs = ['S*-MM-UN']
algs = ['S*-MM']
# algs = ["S*-BS"]

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
