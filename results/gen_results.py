"""this file is used to generate spreadsheet results"""
import itertools as it
import random
import pickle
import os
import argparse

# for deterministic behavior
random.seed(123)

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

l.handlers[1].addFilter(CustomFilter())

from steinerpy.environment import EnvType, EnvLoader 
from steinerpy.library.pipeline import GenerateHeuristics
from steinerpy.library.pipeline import GenerateResultsMulti, GenerateBaseLine, Process
import  steinerpy.config as cfg

# sc map names
sc_names = ["WheelofWar.map", "Archipelago.map", "BigGameHunters.map", "Brushfire.map", "Sandstorm.map"]

# mapf map names
mapf_names = ["empty-48-48.map", "brc202d.map", "den520d.map", "lak303d.map", "maze-128-128-10.map", "orz900d.map"]
# mapf_names = ["brc202d.map"]

# 3d map names
grid3d_names = ["Simple.3dmap","Complex.3dmap","DB1.3dmap"]


# load map and preprocessed heuristic
map_names = []
map_names.extend(mapf_names)
map_names.extend(sc_names)
map_names.extend(grid3d_names)

algs = ["S*-BS", "S*-HS", "S*-MM", 
        "S*-BS-UN", "S*-HS-UN", "S*-MM-UN", 
        "S*-MM-LP"] 

# number of terminals
terminals = [10, 20, 30, 40, 50]
h_vals = [1, 0.75, 0.50, 0.25, 0]
# h_vals = [1]
instances = 25

# keep track of previous terminal number and map

starting = True
prev_term_map = []

# change global heuristic to preprocess type
GenerateHeuristics.preload_type="CDH"
cfg.Algorithm.sstar_heuristic_type = "preprocess"
cfg.Algorithm.use_bpmx = True

# bounding procedure, number of reaches
cfg.Pipeline.min_reach_pivots = 1
# define pre-run (setup) function
def prerun_func(graph, terminals):
    # This function will reinitialize all bounds
    # will not touch the cdh table though! Just READONLY
    GenerateHeuristics.cdh_compute_bounds(graph, terminals)

# maximum number of processes to use
parser = argparse.ArgumentParser()
parser.add_argument("cores", help="specify the number of cpu cores to use")
args = parser.parse_args()
cfg.Pipeline.max_processes = int(args.cores)

for ndx, (t, m, h) in enumerate(it.product(terminals, map_names, h_vals)):
    print("no.: ",ndx+1, " num_terms: ", t, "map name: ", m, "h-weight: ", h)

    # load graph (FOR SC maps, name must include "sc/")
    if m in mapf_names:
        graph = EnvLoader.load(EnvType.MAPF, m)
    elif m in sc_names:
        graph = EnvLoader.load(EnvType.GRID_2D, os.path.join("sc", m))
    else:
        # must be 3d 
        graph = EnvLoader.load(EnvType.GRID_3D, m)

    # load preprocessed heuristics only when the  map changes
    if m not in prev_term_map:
        with open("./heuristics/h_"+m + ".pkl", 'rb') as f:
            hdata = pickle.load(f)
        # dont keep regenerating the cdh table, it's slow! 
        GenerateHeuristics.load_results(results=hdata)

    # change heuristic weighting
    cfg.Algorithm.hFactor = h

    ############ generate baseline ##############
    # make sure we reuse the same terminals for the same map/heuristic values
    bl_path = "{}_{}t_{}i_baseline.pkl".format(m, t, instances)
    gen_bs = GenerateBaseLine(graph=graph, save_path=bl_path, file_behavior="SKIP", load_from_disk=True)    # just added load from disk
    # generate random instances only when map or terminals changes
    if not prev_term_map or t not in prev_term_map or m not in prev_term_map:
        gen_bs.randomly_generate_instances(num_of_inst=instances, num_of_terms=t)
    # run
    gen_bs.run()
    # retrieve specific instances
    sp_instances = gen_bs.instances

    ############# generate results ###############
    res_path = "{}_{}t_{}i_{}h_results.pkl".format(m, t, instances, h)
    gen_res = GenerateResultsMulti(graph=graph, save_path=res_path, algs_to_run=algs, file_behavior="SKIP", load_from_disk=True, pre_run_func=prerun_func)
    # specify instances
    gen_res.input_specifed_instances(sp_instances)
    # run generator
    gen_res.run()

    ############ process results #################
    gen_proc = Process(save_path="{}_{}t_{}i_{}h_processed.xlsx".format(m, t, instances, h), file_behavior="SKIP")
    # specify baseline and result files
    gen_proc.specify_files(bl_path, res_path)
    # run to process
    gen_proc.run()


    ##### keep track of map and terminal number
    prev_term_map = [t, m]
