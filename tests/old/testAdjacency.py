#!/usr/bin/env python3

from graphs.graph import MyGraph
import matplotlib.pyplot as plt
from search.search_algorithms import DepthFirstSearch
import pdb

# This is returned from algorithm (a parent list)
testDictCycle = {1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:5}
testDict1 = {1:2, 3:2}
testDict2 = {(5,9):(9,9), (9,-3):(9,9)}

# # Create UNDIRECTED edge dict with weights (THIS IS EASIER)
testDict = testDictCycle
edgeDict = {}
for key in testDict.keys():
    edgeDict[(key, testDict[key])] = 1.0
    edgeDict[(testDict[key], key)] = 1.0

#edgeDict = {(1,2):1, (2,3):1 , (2,4):1, (4,5):1, (4,6):1}
# edgeDict = {(1,2):1, (2,3):1 , (2,4):1, (4,5):1, (5,6):1, (5,7):1}

# create graph using my customized library
graph = MyGraph(edgeDict)

# # test cycle_detection
# isCycle = cycle_detection(graph, 1, visualize=True)
# print("Cycles detected: ",isCycle)

# test cycle (cleaner code)
# TODO test cycle with new method
# SA = DepthFirstSearch(graph, 1, visualize=True)
SA = DepthFirstSearch(graph, 1, visualize=True)
isCycle = SA.use_algorithm()
print("Cycles detected: ",isCycle)

plt.show()





