"""this file is used to generate spreadsheet results"""
import itertools as it
import random

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

# map names
# names = ["WheelofWar.map", "Archipelago.map", "BigGameHunters.map", "Brushfire.map", "Sandstorm.amp"]

# load map and preprocessed heuristic
map_names = ["room-32-32-4.map"]

algs = ["S*-BS", "S*-HS", "S*-MM", 
        "S*-BS-UN", "S*-HS-UN", "S*-MM-UN", 
        "S*-HS-LP", "S*-MM-LP",
        "S*-HS-UN-LP", "S*-MM-UN-LP"] 

# number of terminals
terminals = [20, 40, 60, 80, 100]
h_vals = [0, 0.25, 0.50, 0.75, 1]
instances = 20

# keep track of previous terminal number and map
prev_term_map = None

# change global heuristic to preprocess type
cfg.Algorithm.sstar_heuristic_type = "preprocess"

for ndx, (t, m, h) in enumerate(it.product(terminals, map_names, h_vals)):
    print("no.: ",ndx, " num_terms: ", t, "map name: ", m, "h-weight: ", h)

    # load graph (FOR SC maps, name must include "sc/")
    graph = EnvLoader.load(EnvType.MAPF, m)

    # load preprocessed heuristics
    GenerateHeuristics.load_results(db_location="heuristics/"+m)

    # change heuristic weighting
    cfg.Algorithm.hFactor = h


    ############ generate baseline ##############
    bl_path = "{}_{}t_baseline.pkl".format(m, t)
    gen_bs = GenerateBaseLine(graph=graph, save_path=bl_path, file_behavior="SKIP")
    # generate random instances only when map or terminals changes
    if prev_term_map is None or t not in prev_term_map or m not in prev_term_map:
        gen_bs.randomly_generate_instances(num_of_inst=instances, num_of_terms=t)
    # run
    gen_bs.run()
    # retrieve specific instances
    sp_instances = gen_bs.instances

    ############# generate results ###############
    res_path = "{}_{}t_{}h_results.pkl".format(m, t, h)
    gen_res = GenerateResultsMulti(graph=graph, save_path=res_path, algs_to_run=algs)
    # specify instances
    gen_res.input_specifed_instances(sp_instances)
    # run generator
    gen_res.run()

    ############ process results #################
    gen_proc = Process(save_path="{}_{}t_{}h_processed.xlsx".format(m, t, h), file_behavior="OVERWRITE")
    # specify baseline and result files
    gen_proc.specify_files(bl_path, res_path)
    # run to process
    gen_proc.run()


    ##### keep track of map and terminal number
    prev_term_map = [t, m]
