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

# loading info
# loader = [(EnvType.MAPF, lambda x: x),
#           (EnvType.GRID_2D, lambda x: os.path.join("sc", x)), 
#         (EnvType.GRID_3D, lambda x: x),] 
loader = [(EnvType.GRID_3D, lambda x: x),] 

instances = []

# mapf maps 
names = ["brc202d.map", "den520d.map", "lak303d.map", "maze-128-128-10.map", "orz900d.map"]
# names = ["maze-128-128-10.map"]

size_scale = [1, 1, 1, 1, 1]
plim =  [128, 128, 128, 128, 128]
slim = [64, 64, 64, 64, 64]

# instances.append((names, size_scale, plim, slim))

# cfg.Pipeline.pivot_limit = 128
# cfg.Pipeline.subset_limit = 64
# load sc grid2d maps
names = ["WheelofWar.map", "Archipelago.map", "BigGameHunters.map", "Brushfire.map", "Sandstorm.map"]
# names = ["Archipelago.map"]
# names = ["WheelofWar.map"]
# names = ["Archipelago.map"]
# names = ["den520d.map"]
# names = ["room-32-32-4.map"]

size_scale = [1, 1, 1, 1, 1]
plim = [256, 256, 256, 256, 256]
slim = [16, 16, 16, 16, 16]

# instances.append((names, size_scale, plim, slim))


# cfg.Pipeline.pivot_limit = 256
# cfg.Pipeline.subset_limit = 16

# try memory limited dijkstra (real slow prob) (sc)
# names = ["Caldera.map"]

# 3d map
# names = ["A1.3dmap"]
# names = ["Simple.3dmap"]
# names = ["FA1.3dmap"]
# names = ["Complex.3dmap"]
# names = ["FA1.3dmap", "Complex.3dmap", "Simple.3dmap"]
# names = ["Complex.3dmap", "Simple.3dmap"]
# names = ["DB1.3dmap"]

# cfg.Pipeline.pivot_limit = 75

#mapf
# names = ["den312d.map"]

# names = ["Simple.3dmap","Complex.3dmap","DB1.3dmap"]
# size_scale = [1, 8, 32]
# plim =  [256, 256, 256]
# slim = [4, 4, 4]

names = ["DB1.3dmap"]
size_scale = [32]
plim =  [256]
slim = [4]
instances.append((names, size_scale, plim, slim))

stats = []
# for n in names:
# for n, c in zip(names, size_scale):
for ndx, instance in enumerate(instances):

    # prepare loader variables
    l = loader[ndx]

    for n, c, p, s in zip(*instance):
        # name, scale factor, pivot amounts, subset amounts, loader info
        print("preprocessing: {}".format(n))
        
        # load graph
        graph = EnvLoader.load(l[0], l[1](n))

        # set memory limits
        cfg.Pipeline.node_limit = int(graph.node_count()/c)
        cfg.Pipeline.pivot_limit = p
        cfg.Pipeline.subset_limit = s
        
        # time generation and saving of heuristic files
        t1 = timer()
        res = GenerateHeuristics.gen_and_save_results(graph, save_path=os.path.join("heuristics", "h_"+n+".pkl"), file_behavior="OVERWRITE")
        t2 = timer()

        
        print("time to generate h-{}: {}".format(n, t2-t1))
        stats.append((n, c, p, s, t2-t1))

    # # t1 = timer()
    # # for 3d grid map
    # graph = EnvLoader.load(EnvType.GRID_3D, n)
    # # for sc maps
    # # graph = EnvLoader.load(EnvType.GRID_2D, os.path.join("sc", n))
    # # for mapf maps
    # # graph = EnvLoader.load(EnvType.MAPF, n)

    # # set memory limit of cdh table (for mapf graphs 16|V|, sc maps |V|/10)
    # cfg.Pipeline.node_limit = int(graph.node_count()/c)
    # # cfg.Pipeline.node_limit = int(graph.node_count())
    # # cfg.Pipeline.node_limit = int(graph.node_count()/2)
    # # for Complex map
    # # cfg.Pipeline.node_limit = int(graph.node_count()/16)
    # # for simple map
    # # cfg.Pipeline.node_limit = int(graph.node_count()/2)
    # # for DB1 map
    # # cfg.Pipeline.node_limit = int(graph.node_count()/100)
    # # for FAI map
    # # cfg.Pipeline.node_limit = int(graph.node_count()/64)
    # # cfg.Pipeline.node_limit = int(graph.node_count()/410)

    # # graph.show_grid()
    # # to actually gen and save heuristics as .sqlite db
    # # res = GenerateHeuristics.gen_and_save_results(graph, save_path=os.path.join("heuristics", "h_"+n+".sqlite"))

    # # gen/save heuristics as .pkl file
    # t1 = timer()
    # res = GenerateHeuristics.gen_and_save_results(graph, save_path=os.path.join("heuristics", "h_"+n+".pkl"), file_behavior="OVERWRITE")

    # # give namespace to redis database
    # # res = GenerateHeuristics.gen_and_save_results(graph, save_path="{}.redis".format(n))
    # t2 = timer()

    # print("time to generate h-{}: {}".format(n, t2-t1))
    # stats.append((n, t2-t1))

for s in stats:
    print(s)
