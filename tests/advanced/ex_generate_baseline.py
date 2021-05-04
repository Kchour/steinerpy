import cloudpickle
import random as rd
import os

import steinerpy.config as cfg
cfg.Animation.visualize = False

from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.algorithms.kruskal import Kruskal

# Spec out our squareGrid
minX = -15			# [m]
maxX = 15           
minY = -15
maxY = 15
grid = None         # pre-existing 2d numpy array?
grid_size = 1       # grid fineness[m]
grid_dim = [minX, maxX, minY, maxY]
n_type = 8           # neighbor type

# Create a squareGrid using GraphFactory
graph = GraphFactory.create_graph("SquareGrid", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type)      

# generate 5 run instances with N terminals
N = 5
instances = 5

def while_func():
    array = set()
    while len(array) < N:
        array.add((rd.randint(minX, maxX), rd.randint(minY, maxY)))
    yield array
terminals = [[list(i) for i in while_func()][0] for j in range(instances)]

# Run Kruskals on each and save results
solution = []
for t in terminals:
    ko = Kruskal(graph, t)
    ko.run_algorithm()
    solution.append(ko.return_solutions())

# dump instances, and solution
directory = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(directory, 'baseline.pkl'), 'wb') as f:
    cloudpickle.dump({
        'terminals': terminals,
        'solution': solution
    }, f)

print("wip")

