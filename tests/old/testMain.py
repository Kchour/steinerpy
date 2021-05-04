#!/usr/bin/env python3
import numpy as np

from graphs.ogm import OccupancyGridMap
from graphs.graph import SquareGrid, MyGraph

from search.search_algorithms import AStarSearch, DepthFirstSearch
from search.search_utils import reconstruct_path, path_length

from animation.animation import Animate, SaveAnimation

import matplotlib.pyplot as plt

import copy
import random as rd
from codetiming import Timer
import pickle

#Some options
saveVideo = False
doVisualize = True
saveVideoName = "test_20x20"

saveName = "test"
saveGraph = False
loadGraph = False

# Define Grid
grid_size = 1       # [m]
minX = -10			# [m]
maxX = 10
minY = -10
maxY = 10
grid_dim = np.array([minX, maxX, minY, maxY])

''' Either load previous data or generate new one '''
if loadGraph:
    terminals = pickle.load( open( "./results/pickle/terminals_" + saveName + ".p", "rb" ) )
    obstacleList = pickle.load( open( "./results/pickle/obstacles_" + saveName + ".p", "rb" ) )
else:
    # Define an occupancy square grid map 
    obstacleList = [(9,5),(5,7),(-2.5,0)]
    #obstacleList = [(rd.randint(minX,maxX), rd.randint(minY, maxY)) for z in range(5)]
    #obstacleList = []

    # Define Terminals and plot
    terminals = [(-9,-9), (9,9), (-9,9), (5,9), (9,-3), (0,0),(6,5), (-7,4), (0,4), (-4,0)] 
    #terminals = [(rd.randint(minX,maxX), rd.randint(minY, maxY)) for z in range(10) ]
    #terminals = [(-9,-9), (-9,0), (-9,9)]

# Define OGM and square grid map, where each node is either 1 or 0, with 8 neighbors
ogm = OccupancyGridMap(grid_size, grid_dim, np.array(obstacleList))   #requires np.array
graph = SquareGrid(grid=ogm.grid, grid_dim=grid_dim, grid_size=grid_size, type_=8)

''' Use helper class to animate things (iteratively plot things'''
# Plotting Terminals
if doVisualize:
    plotTerminals = Animate(1, (minX, maxX), (minY, maxY), grid_size, 5, 'o', 10, 0, 10)
    plotTerminals.update(np.array(terminals).T.tolist())

# plotting obstacles
if obstacleList and doVisualize:
    plotObstacles = Animate(1, (minX, maxX), (minY, maxY), grid_size, 5, 'o', 10, 0, 10)
    plotObstacles.update(np.array(obstacleList).T.tolist())

# Plotting length of added segment
# plotLength = Animate(2, None, None, grid_size, 5, 'o', 10, 0, 10)
plotLengthData = [[],[]]

# use helper class to save animations
if saveVideo and doVisualize:
    save_animation = SaveAnimation(1, saveVideoName+'.mp4')

'''Store the instance of Search Algorithm classes'''
components = {}
for index, item in enumerate(terminals):
    SA = AStarSearch(graph, item, None, h_type='zero', visualize=doVisualize)
    components.update({(index): {'terminal':item, 'object' :SA, 'results': {}}})
    
''' Construct the Steiner Tree '''
solutionForest = {}               # solution containing our Steiner Tree (set S)
shortestPathTerminals = {}        # (i,j)   indicates whether there is existed a prior shortest path between terminals i,j
numEdges = 0                      # number of edges in our solution. Termination condition
edgeDict = {}                     # Used to check for cycles
totalCosts = 0                    # store total path length of tree

# Time the algorithm
t = Timer(text="Elapsed time: {milliseconds:.0f} ms")
t.start()


# STEP 5: Return to step 1
while (numEdges<len(terminals)-1):

    # STEP 1:   Each component nominates a node and we store the outputs into a hash table
    for c1_key, c1_val in components.items():
        parent, g, frontier, current = c1_val['object'].iterate_algorithm()
        if current != None:

            # STEP 2: (a bit different) update closed lists, open lists, linked lists, and current node for c1
            c1_val['results'] = {'closedList': g, 'frontier':frontier, 'parentList': parent, 'lastClosedNode':  current}

            # STEP 3: Check for overlaps in closed list, indicates a shortest path, Otherwise return to step 1
            for c2_key, c2_val in components.items():
                if c1_key != c2_key and c2_val['results'] and (c1_key, c2_key) not in shortestPathTerminals:   

                    # STEP 4: check latest expanded node, see which pair of terminals it belongs to
                    if c1_val['results']['lastClosedNode'] in c2_val['results']['closedList'].keys():
                        # indicate that a shortest path was previously found. So don't repeat
                        shortestPathTerminals[(c1_key, c2_key)] = 1
                        shortestPathTerminals[(c2_key, c1_key)] = 1
                        print('\n Intersection! ', (c1_key, c2_key))

                        # Create undirected dict of edges
                        edgeDict[(c1_val['terminal'], c2_val['terminal'])] = 1.0
                        edgeDict[(c2_val['terminal'], c1_val['terminal'])] = 1.0
                        mygraph = MyGraph(edgeDict)

                        # Check for Cycles (ONLY ON TERMINALS!)
                        SA = DepthFirstSearch(mygraph, c1_val['terminal'], visualize=False)
                        isCycle = SA.use_algorithm()   
                        print("Cycle Detection: ",isCycle)

                        if isCycle:
                            edgeDict.pop((c1_val['terminal'], c2_val['terminal']))
                            edgeDict.pop((c2_val['terminal'], c1_val['terminal']))

                        else:      
                            numEdges +=1
                            print("Added edge number: ", numEdges)

                            #Path reconstruction 
                            pathA = reconstruct_path(c1_val['results']['parentList'], c1_val['terminal'], \
                                                    c1_val['results']['lastClosedNode'],'forward')
                            pathB = reconstruct_path(c2_val['results']['parentList'], c2_val['terminal'], \
                                                    c1_val['results']['lastClosedNode'],'reverse')
                            path = np.vstack((pathA, pathB))
                            
                            # Add to our solution set
                            solutionForest.update({(c1_key, c2_key): {'path':path, 'cost':path_length(path) }})
                            
                            # Show length of edge added
                            totalCosts += path_length(path)
                            print("Total and added cost: ", totalCosts, path_length(path))
                            plotLengthData[0].append(len(plotLengthData[0]))
                            plotLengthData[1].append(path_length(path))
                            # for px,py in zip(plotLengthData[0], plotLengthData[1]):
                            #     plotLength.update((px,py))

                            # Plot Path
                            if doVisualize:
                                plotEdges= Animate(1, (minX, maxX), (minY, maxY), grid_size, 5, '-r', 1, 0, 50)
                                for p in path:
                                    plotEdges.update((p[0],p[1]))
    if saveVideo and doVisualize:
        save_animation.update()

timeValue = t.stop()
print("Time Taken: ",timeValue)

print("Lengths Added: ",plotLengthData)

# save animation?
if saveVideo and doVisualize:
    # Add some delay
    t.start()
    t2 = Timer(text="Elapsed time: {milliseconds:.0f} ms")
    t2.start()
    while (t2._start_time - t._start_time < 3):
        save_animation.update()
        t2.stop()
        t2.start()

    save_animation.save()



# Save terminals/objects list
if saveGraph and not loadGraph:
    pickle.dump( terminals, open( ".results/pickle/terminals_" + saveName + ".p", "wb" ) )
    pickle.dump( obstacleList, open( ".results/pickle/obstacles_" + saveName + ".p", "wb" ) )
# Ensure plots don't close at the end
print("Finished")
plt.show()
