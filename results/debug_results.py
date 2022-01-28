"""Debug certain instances"""
import pickle
from steinerpy.library.pipeline import GenerateResultsMulti, GenerateBaseLine, Process
from steinerpy.library.pipeline import GenerateHeuristics
from steinerpy.environment import EnvType, EnvLoader 
import steinerpy.config as cfg

# enable visualization
cfg.Animation.visualize = True

# map_name
map_name = "room-32-32-4.map"
# load baseline pkl
bl_file = "room-32-32-4.map_20t_baseline.pkl"
bl_data = pickle.load(open(bl_file, 'rb'))
# load results pkl
res_file = "room-32-32-4.map_20t-0h_results.pkl"
# select instance
instance_num = 0

# algorithms to run
# algs = ["S*-BS", "S*-HS", "S*-MM", 
#         "S*-BS-UN", "S*-HS-UN", "S*-MM-UN", 
#         "S*-HS-LP", "S*-MM-LP",
#         "S*-HS-UN-LP", "S*-MM-UN-LP"] 
algs = ["S*-BS", "S*-MM"]

# load graphs
graph = EnvLoader.load(EnvType.MAPF, map_name)

# load preprocessed heuristics (.sqlite)
# GenerateHeuristics.load_results(db_location="heuristics/"+"{}.sqlite".format(map_name))
# set heuristic sqlite name
GenerateHeuristics.load_heuristic_name(load_name="heuristics/"+"{}.sqlite".format(map_name))

# set db namespace for redis (not successful)
# GenerateHeuristics.load_results(db_location=map_name+".redis")
# try connecting to redis per processor (not successful)
# GenerateHeuristics.load_heuristic_name(load_name=map_name+".redis")

# change global heuristic to preprocess type
cfg.Algorithm.sstar_heuristic_type = "preprocess"

# get ready to generate results
gen_res = GenerateResultsMulti(graph=graph, algs_to_run=algs)
# specify instances from baseline pkl file (list of list)
gen_res.input_specifed_instances([bl_data['terminals'][instance_num]])
# run generator
res = gen_res.run()
pass

# process into pandas dataframes
new_bl_data = {'terminals': [bl_data['terminals'][instance_num]], 'solution': [bl_data['solution'][instance_num]]}
gen_proc = Process()
gen_proc.specify_data(new_bl_data, res)
df = gen_proc.run()

print(df)
