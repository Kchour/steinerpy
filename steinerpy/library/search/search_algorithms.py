"""Iterative search algorithms with Astar, DepthFirstSearch, and Dijkstra 

Example:

    # Create search object
    graph = GraphFactory(...)
    asearch = AStarSearch(graph, start, goal, h_type, visualize=False)
    
    # Query the search object
    parentList, optimalCosts = asearch.use_algorithm()

    # Run one iteration of search
    parentList, optimalCosts, frontierList, currentNode = asearch.iterate_algorithm()

Todo:
    - Consider combining this class with generic search?
    - Dijkstra class not fully implemented
    - More documentation later

"""

import matplotlib.pyplot as plt
import numpy as np
import time
# import networkx as nx

from steinerpy.library.search.search_utils import PriorityQueue, PriorityQueueHeap
from steinerpy.library.animation.animationV2 import AnimateV2

class Search:
    
    total_expanded_nodes = 0

    def __init__(self, graph, start, goal):
        self.graph = graph
        self.start = start
        self.goal = goal

    def set_start(self, start):
        self.start = start

    def set_goal(self, goal):
        self.goal = goal

    @classmethod
    def update_expanded_nodes(cls):
        cls.total_expanded_nodes +=1

    @classmethod
    def reset(cls):
        """Reset all class variables """
        cls.total_expanded_nodes = 0

class AStarSearch(Search):
    """Standard A-Star search on any generic graph, though heuristics are only defined for grid-based graphs atm"""
    def __init__(self, graph, start, goal, h_type, visualize=False):
        Search.__init__(self, graph, start, goal)
        self.h_type = h_type
        self.visualize = visualize

        # A star initialize openList, closedList
        # self.frontier = PriorityQueue()       # The OPENLIST
        self.frontier = PriorityQueueHeap()
        self.frontier.put(self.start, 0)      # PUT START IN THE OPENLIST
        self.parent = {}              # parent, {loc: parent}
        # g function dict, {loc: f(loc)}, CLOSED LIST BASICALLY
        self.g = {}
        self.parent[self.start] = None
        self.g[self.start] = 0

        if self.goal is not None:
            if type(self.goal) is not set:
                self.set_of_goal = set({self.goal})
            else:
                self.set_of_goal = self.goal
        else:
            self.set_of_goal = set()

    def heuristic(self, a, b, type_='manhattan'):
        """ Grid based heuristics """
        (x1, y1) = a
        (x2, y2) = b
        if type_ == 'manhattan':
            return abs(x1 - x2) + abs(y1 - y2)
        elif type_ == 'euclidean':
            v = [x2 - x1, y2 - y1]
            return np.hypot(v[0], v[1])
        elif type_ == 'diagonal_uniform':
            # Chebyshev Distance
            return max(abs(x1 - x2), abs(y1 - y2))
        elif type_ == 'diagonal_nonuniform':
            dmax = max(abs(x1 - x2), abs(y1 - y2))
            dmin = min(abs(x1 - x2), abs(y1 - y2))
            return 1.414*dmin + (dmax - dmin)
    
    def use_algorithm(self):
        """ Usage:
            - call to runs full algorithm until termination

            Returns:
            - a linked list, 'parent'
            - hash table of nodes and their associated min cost, 'g'
        """

        # Avoid circular imports
        from steinerpy.algorithms.common import Common

        frontier = self.frontier
        parent = self.parent
        g = self.g

        # Ensure searched nodes have been reset
        AStarSearch.reset()

        while not frontier.empty():
            _, current = frontier.get()  # update current to be the item with best priority

            # Update stats logging
            AStarSearch.update_expanded_nodes()

            if self.visualize:
                # Update plot with visuals
                # self.animateCurrent.update(current)
                # if self.graph.obstacles is not None and frontier.empty():
                #     a,b,c,d = self.graph.grid_dim
                #     xlim,ylim = (a,b),(c,d)
                #     AnimateV2.add("obstacles", np.array(self.graph.obstacles).T.tolist(), xlim=xlim, ylim=ylim, markersize=0.5, marker='o', color='k')

                # if np.fmod(self.total_expanded_nodes, 2000)==0:
                AnimateV2.add_line("current", current[0], current[1], markersize=10, marker='o')
                # Animate closure
                AnimateV2.add_line("current_animate_closure", current[0], current[1], markersize=10, marker='o', draw_clean=True)
                AnimateV2.update()


            # # early exit if we reached our goal
            # if current == self.goal:
            #     return parent, g

            # early exit if all of our goals in the closed set
            if self.set_of_goal:
                self.set_of_goal -= set({current}) 
                if len(self.set_of_goal) == 0:
                # if len(set(self.goal).intersection(set(g)-set(frontier.elements))) == len(self.goal):
                    return parent, g
            elif current == self.goal:
                return parent, g

            # expand current node and check neighbors
            neighbors_data = []
            for next in self.graph.neighbors(current):
                g_next = g[current] + self.graph.cost(current, next)
                # if next location not in CLOSED LIST or its cost is less than before
                # Newer implementation
                if next not in g or g_next < g[next]:
                    g[next] = g_next
                    if self.h_type == 'zero' or self.goal == None or self.h_type is None:
                        priority = g_next 
                    else:
                        # priority = g_next + self.heuristic(self.goal, next, self.h_type)
                        priority = g_next + Common.grid_based_heuristics(type_=self.h_type, next=next, goal=self.goal)
                    frontier.put(next, priority)
                    parent[next] = current
                    neighbors_data.append(next)

            if self.visualize:
                # # self.animateNeighbors.update(next)
                # if np.fmod(self.total_expanded_nodes, 100000)==0 or self.total_expanded_nodes == 0:

                data = [k[2] for k in self.frontier.elements.values()]
                if data:
                    AnimateV2.add_line("frontier", np.array(data).T.tolist(), markersize=8, marker='D', draw_clean=True)
                    AnimateV2.update()

                # AnimateV2.add("neighbors", np.array(neighbors_data).T.tolist(), markersize=8, marker='D', draw_clean=True)
                # AnimateV2.update()


        return parent, g

class DepthFirstSearch(Search):
    """ Depth first search implemented with a LIFO structure. Done by successively
        increasing "priority" (more negative) in our priorityQueue

    Todo:
        - Add each path to a list!
        - This is a super rough way to do this

    """
    def __init__(self, graph, start, visualize=False):
        Search.__init__(self,graph, start, None)
        self.visualize = visualize

        self.frontier = PriorityQueue()
        self.frontier.put(self.start, 0)
        self.parent = {}
        self.parent[self.start] = None
        self.closedList = []    #leaf nodes

        if visualize:
            self.fig, self.ax = plt.subplots(1,1, num=10)
            plt.show(block=False)
            self.nxG = nx.Graph()
            self.pos = []

    def use_algorithm(self):
        frontier = self.frontier
        parent = self.parent
        g_next = 0
        visited = []
        visitedNodes = []
        isCycle = False
        
        while not frontier.empty():
            _, current = frontier.get()
            #ensures we expand unvisited nodes!                 
            visited.append(current)
            neighbors = self.graph.neighbors(current)
            for next in neighbors:
                g_next -= 1
                if next not in visited:
                    priority = g_next
                    frontier.put(next, priority)
                    parent[next] = current
                    # print(current, next, visited)

                    # record nodes visited
                    #visitedNodes1.append(current)
                    visitedNodes.append(next)

                    if self.visualize:
                        self.nxG.add_edge(current, next)
                        if not self.pos:
                            self.pos = nx.spring_layout(self.nxG)
                        else:
                            self.pos = nx.spring_layout(self.nxG, pos =self.pos, fixed=visited)
                        self.ax.clear()
                        nx.draw(self.nxG, ax=self.ax, pos=self.pos, edge_color="blue", with_labels = True)
                        self.ax.set_xticks([])
                        self.ax.set_yticks([])
                        plt.pause(0.0001)
        # We can check for duplicates in here
        if len(visitedNodes) != len(set(visitedNodes)):
           isCycle = True
        return isCycle


