from timeit import default_timer as timer
import os
import steinerpy.config as cfg

from steinerpy.environment import EnvType, EnvLoader 
from steinerpy.library.pipeline import GenerateHeuristics

### Debugging ###

# name = "sc/Sandstorm.map"
# graph = EnvLoader.load(EnvType.GRID_2D, name)

# name = "room-32-32-4.map"
# graph = EnvLoader.load(EnvType.MAPF, name)

#################

# mapf maps 

# load sc grid2d maps
names = ["WheelofWar.map", "Archipelago.map", "BigGameHunters.map", "Brushfire.map", "Sandstorm.map"]
# names = ["Archipelago.map"]
# names = ["den520d.map"]
# names = ["room-32-32-4.map"]

# try memory limited dijkstra (real slow prob) (sc)
# names = ["Caldera.map"]

# 3d map
# names = ["A1.3dmap"]
names = ["Simple.3dmap"]

for n in names:
    print("preprocessing: {}".format(n))
    t1 = timer()
    # for 3d grid map
    graph = EnvLoader.load(EnvType.GRID_3D, n)
    # for sc maps
    # graph = EnvLoader.load(EnvType.GRID_2D, os.path.join("sc", n))
    # for mapf maps
    # graph = EnvLoader.load(EnvType.MAPF, n)
    
    # to actually gen and save heuristics as .sqlite db
    # res = GenerateHeuristics.gen_and_save_results(graph, save_path=os.path.join("heuristics", "h_"+n+".sqlite"))

    # gen/save heuristics as .pkl file
    res = GenerateHeuristics.gen_and_save_results(graph, save_path=os.path.join("heuristics", "h_"+n+".pkl"), file_behavior="OVERWRITE")

    # give namespace to redis database
    # res = GenerateHeuristics.gen_and_save_results(graph, save_path="{}.redis".format(n))
    t2 = timer()

    print("time to generate h-{}: {}".format(n, t2-t1))