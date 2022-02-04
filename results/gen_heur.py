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

cfg.Pipeline.pivot_limit = 100

# try memory limited dijkstra (real slow prob) (sc)
# names = ["Caldera.map"]

# 3d map
# names = ["A1.3dmap"]
# names = ["Simple.3dmap"]
names = ["FA1.3dmap"]
# names = ["Complex.3dmap"]
# names = ["FA1.3dmap", "Complex.3dmap", "Simple.3dmap"]

cfg.Pipeline.pivot_limit = 4

#mapf
# names = ["den312d.map"]


stats = []
for n in names:
    print("preprocessing: {}".format(n))
    # t1 = timer()
    # for 3d grid map
    graph = EnvLoader.load(EnvType.GRID_3D, n)
    # for sc maps
    # graph = EnvLoader.load(EnvType.GRID_2D, os.path.join("sc", n))
    # for mapf maps
    # graph = EnvLoader.load(EnvType.MAPF, n)

    # set memory limit of cdh table (for sc graphs 16|V|, 3d, 0.5|V|)
    # cfg.Pipeline.node_limit = int(16*graph.node_count())
    # for Complex map
    cfg.Pipeline.node_limit = int(graph.node_count()/64)
    # for FAI map
    # cfg.Pipeline.node_limit = int(graph.node_count()/64)

    # graph.show_grid()
    # to actually gen and save heuristics as .sqlite db
    # res = GenerateHeuristics.gen_and_save_results(graph, save_path=os.path.join("heuristics", "h_"+n+".sqlite"))

    # gen/save heuristics as .pkl file
    t1 = timer()
    res = GenerateHeuristics.gen_and_save_results(graph, save_path=os.path.join("heuristics", "h_"+n+".pkl"), file_behavior="OVERWRITE")

    # give namespace to redis database
    # res = GenerateHeuristics.gen_and_save_results(graph, save_path="{}.redis".format(n))
    t2 = timer()

    print("time to generate h-{}: {}".format(n, t2-t1))
    stats.append((n, t2-t1))

for s in stats:
    print(stats)