from steinerpy.library.graphs.parser import DataParser
from steinerpy import Context
import os

# Some online_data options
# steinlib/B/
# simple/test.stp

###############################
###  RUN STEINLIB INSTANCES ###
###############################

# # get online data
# # filename = "steinlib/B/b18.stp"
# filename = "simple/test.stp"
# cwd = os.path.dirname(os.path.realpath(__file__))
# filepath = os.path.join(cwd, filename)

# # Use DataParser to generate a graph object
# g, T = DataParser.parse(filepath, dataset_type="steinlib")


###############################
###    RUN MAPF INSTANCES   ###
###############################
# get online data and define terminals
# filename, T = "mapf/Berlin_1_256.map", [(197, 149), (60, 87), (39, 130), (143, 160), (94, 222)]  
# filename, T = "mapf/maze-32-32-2.map", [(18, 5), (9, 11), (26, 9), (25, 15), (6, 17)]  
filename, T = "mapf/room-64-64-16.map", [(22, 47), (15, 9)]
cwd = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(cwd, filename)

# Use DataParser to generate a graph object
g = DataParser.parse(filepath, dataset_type="mapf")
g.show_grid()
#######################################
###  USE CONTEXTUALIZER TO RUN ALGS ###
#######################################

# Create context
context = Context(g, T)

context.run('Kruskal')
results1 = context.return_solutions()

context.run('SstarAstar')
results2 = context.return_solutions()

print("Finished Parse Test")
