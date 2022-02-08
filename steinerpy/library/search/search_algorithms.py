"""This module provides a generic incremental search class, that breaks up nomination and update phase

    TODO: rename fcosts_func to priority_func

"""
from steinerpy.library.graphs.graph import IGraph
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import logging
from typing import Iterable, List
import random

import steinerpy.config as cfg
from steinerpy.library.animation import AnimateV2
from steinerpy.library.misc.utils import MyTimer
from steinerpy.library.search.search_utils import PriorityQueueHeap
from steinerpy.library.search.search_utils import DoublyLinkedList

from steinerpy.heuristics import Heuristics

# create logger
my_logger = logging.getLogger(__name__)


def CostFuncInterface(this, cost_so_far, next):
    """User can pass in a different pCostFunc during instantiation of GenericSearch
    
        Normally, fcost(n) = gcost(n) + hcost(n, goal), but this function 
        can be used very generically to define the priority of node 'next'        
    
        User must keep track of fcosts! := g + h

    Parameters:
        this (GenericSearch): The search object that called this p function
        cost_so_far (dict): Contains all nodes with finite g-cost
        next (tuple): The node in the neighborhood of 'current' to be considered 

    Returns:
        priority (float): The priority value for the node 'next'

    """
    # Must keep track of Fcosts
    this.f[next] = cost_so_far[next]
    #
    # ...
    #
    #
    # return some_float_value


class Search:
    """ Base Class `Search` can be extended by any iterative search algorithm.

    Generic search algorithm with open, closed, and linked lists. The user can pass in custom g,h functions 

    Parameters:
        graph (SquareGrid, MyGraph): An object from grid_utils/graph module.
            "SquareGrid" vertices are defined by 2D tuples (x_i, y_i) 
            "MyGraph"    vertices are defined N-dim tuples (z_1, ... , z_n). A generalized graph!
        start (tuple): Start vertex, `tuple` must belong to class <graph>.
        goal (tuple): End vertices, keyed by id. May be None
            For use with `Framework` class, must be an iterable (i.e. list, tuple, set) or `dict`
        frontierType (PriorityQueue()): The Open List or frontier class type, implemented as a class from search/search_utils/  
            (PriorityQueue, PriorityQueueHeap). A priority queue returns the item with min priority value.
        pCostsFunc: A function returning the pCosts of node u. Returns a scalar value. Arguments are (self, glist, next_node)
        id (tuple): An ID number for the current search object. Optional if not using `Framework`.

    Attributes:
        id: set by the parameter `id`
        graph: set by parameter `graph`
        start: set by parameter `start`
        goal: set by parameter `goal`
        current (tuple): The most recently expanded node, i.e. just added to the closed list
        frontier (frontierType): set by parameter `frontierType`
        g (dict): The cost_so_far dict in the form {v: value}, where "v" is a vertex/node and "value" is the gCost(v).
        parent (dict): A linked list (implemented with a dict), in the form {v: parent(v)}, where parent(v), is the parent node of "v". 
            Note that "v" is member of `graph`.
        fCosts (fCostsFunc): Defined by input parameter fCostsFunc, should return a scalar
        nominated (bool): True if our current search object has nominated a node, ensures we do not
            nominate more than once at a time
        closedList (dict): Consist of all previously expanded nodes, a dict {closed(v_i): fCosts}. Not necessary
        currentF (float): The Fcost of `current`
        minR (float): minimum radius of our set, by observing the boundary nodes. Not necessary atm
        currentGoal (tuple): The closest reachable goal

    Variables:
        total_closed_nodes (int):
        total_opened_nodes (int):
        total_expanded_nodes (int): Counter that is incremented when we expand a node from the open set
    """
    total_expanded_nodes = 0
    def __init__(self, graph, start, goal):
        # ================ Required Definitions ===================== #
        self.graph = graph                         
        self.start = start
        self.goal = goal     
        self.current = None

        # Open List
        self.frontier = PriorityQueueHeap()            
        self.frontier.put(start, 0)
        
        # The cost so far (includes both frontier and closed list)
        # TODO: May have to modify g merging in the future for primal-dual
        self.g = {}
        self.g[start] = 0

        # Linked List, used to identify parents of a node along a path
        self.parent = {}
        self.parent[start] = None

        # F costs function object for priority updates

    ################################################
    ###   Class methods for updating some stats  ###
    ################################################

    @classmethod
    def update_expanded_nodes(cls):
        cls.total_expanded_nodes +=1
    
    @classmethod
    def reset(cls):
        """Reset all class variables """
        cls.total_expanded_nodes = 0

    def reconstruct_path(self, goal, start=None, order='forward'):
        '''Given a linked list, rebuild a path back from a goal node 
            to a start (or root node, if start is not specified)

        paremeters:
            parents: removed. kept for compatibility reasons
            start: a tuple (x,y) position. Optional
            goal: a tuple (x,y) position. Mandatory
            order: 'forward', or 'reverse' 

        Attributes:
            parent: a singly-linked list using python dict

        '''
        current = goal
        path = []
        while current != start and current!= None:
            # Detect cycles and break out of them
            if current in path:
                # print("WARNING CYCLE DETECTED")
                break
            path.append(current)
            #  current = parents[current]
            current = self.parent.get(current, None)

        if start != None:
            path.append(start)
        if order == 'forward':
            path.reverse()

        return path

class UniSearch(Search):

    def __init__(self, graph, start, goal, heuristic_type="zero", visualize=False, stopping_critiera=None, **kwargs):
        """If heuristic_type is part of a grid, then cfg.Algorithm.graph_domain
        must be set to "grid"


        """
        Search.__init__(self, graph, start, goal)
        self.heuristic_type = heuristic_type
        self.visualize = visualize

        # A star initialize openList, closedList
        self.frontier = PriorityQueueHeap()
        self.frontier.put(self.start, 0)      # PUT START IN THE OPENLIST
        self.parent = {}              # parent, {loc: parent}

        # both the closed and open set shared in g.
        self.g = {}
        self.parent[self.start] = None
        self.g[self.start] = 0

        # Allow for multiple goals, as in the case of dijkstra
        if self.goal is not None:
            if type(self.goal) is not set:
                self.set_of_goal = set({self.goal})
            else:
                self.set_of_goal = self.goal
        else:
            self.set_of_goal = set()

        # stopping criteria for early termination
        self.stopping_criteria = stopping_critiera 

        # additional kwargs for stopping criteria aside from self
        self.kwargs = kwargs

    def use_algorithm(self):
        """Run algorithm until termination

            Returns:
            - a linked list, 'parent'
            - hash table of nodes and their associated min cost, 'g'
        """
        # Ensure searched nodes have been reset
        UniSearch.reset()
        if self.visualize:
            # reset figure between runs
            AnimateV2.delete("current")
            AnimateV2.delete("current_animate_closure")
            AnimateV2.delete("frontier")

        while not self.frontier.empty():
            _, current = self.frontier.get()
            self.current = current


            # Update stats logging
            UniSearch.update_expanded_nodes()

            # Update stats
            if self.visualize:
                # if np.fmod(self.total_expanded_nodes, 2000)==0:
                AnimateV2.add_line("current", current[0], current[1], markersize=10, marker='o')
                # Animate closure
                AnimateV2.add_line("current_animate_closure", current[0], current[1], markersize=10, marker='o', draw_clean=True)
                AnimateV2.update()

            # early exit if all of our goals in the closed set
            if self.set_of_goal:
                self.set_of_goal -= set({current}) 
                if len(self.set_of_goal) == 0:
                    return self.parent, self.g
            elif current == self.goal:
                return self.parent, self.g

            # custom stopping criteria
            if self.stopping_criteria is not None:
                if self.stopping_criteria(self, **self.kwargs):
                    return self.parent, self.g


            # expand current node and check neighbors
            neighbors_data = []
            for next in self.graph.neighbors(current):
                g_next = self.g[current] + self.graph.cost(current, next)
                # if next location not in CLOSED LIST or its cost is less than before
                # Newer implementation
                if next not in self.g or g_next < self.g[next]:
                    self.g[next] = g_next
                    if self.heuristic_type == 'zero' or self.goal == None or self.h_type is None:
                        priority = g_next 
                    else:
                        priority = g_next + Heuristics.grid_based_heuristics(type_=self.heuristic_type, next=next, goal=self.goal)
                    self.frontier.put(next, priority)
                    self.parent[next] = current
                    neighbors_data.append(next)

            if self.visualize:
                # # self.animateNeighbors.update(next)
                # if np.fmod(self.total_expanded_nodes, 100000)==0 or self.total_expanded_nodes == 0:

                data = [k[2] for k in self.frontier.elements.values()]
                if data:
                    AnimateV2.add_line("frontier", np.array(data).T.tolist(), markersize=8, marker='D', draw_clean=True)
                    AnimateV2.update()


class UniSearchMemLimit(UniSearch):
    """Perform uni directional search

    In linux we can take advantage of lazy memory allocation, so
    creating an initially large np array wont eat up your memory
    until a cell is being written too...


    DEPRECATED
    
    """
    def __init__(self, graph, start, goals):
        # super().__init__(graph, start, goal, heuristic_type, visualize, stopping_critiera, **kwargs)

        # store graph
        self.graph = graph
        # graph dims
        if len(start) == 2:
            # 3d graph
            # store cost-to-come, init start state
            self.g = np.full((graph.xwidth, graph.yheight), np.inf)
        elif len(start) == 3:
            # 3d graph
            # store cost-to-come, init start state
            self.g = np.full((graph.x_len, graph.y_len, graph.z_len), np.inf)

        self.g[start] = 0
        # store frontier, init start state
        self.frontier = PriorityQueueHeap()
        self.frontier.put(start, 0)
        # goal states to stop search early
        self.goals:set = goals

    def use_algorithm(self):
        """Run algorithm until termination

            Returns:
            - a linked list, 'parent'
            - hash table of nodes and their associated min cost, 'g'
        """
        # Ensure searched nodes have been reset
        UniSearchMemLimit.reset()
        t1 = timer()
        while not self.frontier.empty():
            _, current = self.frontier.get()
            self.current = current

            # Update stats logging
            UniSearchMemLimit.total_expanded_nodes += 1

            # early stopping
            if current in self.goals:
                self.goals = self.goals - set(current)
                if not self.goals:
                    break
            
            if UniSearchMemLimit.total_expanded_nodes % 1e3 == 0:
                print("time: ", timer()-t1, "searched nodes: ", UniSearchMemLimit.total_expanded_nodes)

            # print(UniSearchMemLimit.total_expanded_nodes)
            # expand current node and check neighbors
            # neighbors_data = []
            for next in self.graph.neighbors(current):
                g_next = self.g[current] + self.graph.cost(current, next)
                # if next location not in CLOSED LIST or its cost is less than before
                # Newer implementation
                if self.g[next] == np.inf or g_next < self.g[next]:
                    self.g[next] = g_next
                    self.frontier.put(next, g_next)

            # if self.visualize:
            #     # # self.animateNeighbors.update(next)
            #     # if np.fmod(self.total_expanded_nodes, 100000)==0 or self.total_expanded_nodes == 0:

            #     data = [k[2] for k in self.frontier.elements.values()]
            #     if data:
            #         AnimateV2.add_line("frontier", np.array(data).T.tolist(), markersize=8, marker='D', draw_clean=True)
            #         AnimateV2.update()
        

class MultiSearch(Search):
    """The class is used for multi-uni-directional search where multiple search directions (components) are performed (pseudo-)concurrently.
    This is done by ensuring the "cheapest" component is able to update its node via a 'nominate' and 'update' phase.
    Each component must nominate a node (which the user will have to keep track of). Then the user must call
    update() on it. The graph is explored via the nomination and update phase, and shortest paths between components can be discovered.

    This gives the user finer control over the search space, i.e. when to stop, update destinations midway, etc.

    Params:
        frontier_type: Allows user to select the type (a class) of priority queue to use

    Misc:
        visualize (bool): A flag for visualizing the algorithm. Mainly for debug purposes
        animateCurrent (Animate): Animate the current nominated node
        animateClosed (Animate): Animate the history of the closed set
        animateNeighbors (Animate): Animate the open list in the `update`

    Todo:
        * Consider putting animateClosed in the `update` function, because closing does not occur until `update`
    """
    def __init__(self, graph:IGraph, start: List[tuple], goal: Iterable[tuple]=None, pCostsFunc=CostFuncInterface, hCostsFunc=CostFuncInterface, visualize=False, id=None):
        if len(start)==1:
            Search.__init__(self, graph, start[0], goal)
            self.start = start
        else:
            self.graph = graph
            self.start = start
            self.goal = goal
            self.current = None
            # parent creation is delayed during merging
        
        # Visualize algorithm flag
        self.visualize = visualize

        # Keep track of nomination status
        self.nominated = False       # Make sure we don't nominate twice in a row

        # Each search object needs an id
        self.id = (id,)

        # make sure root has the correct starting f value
        # self.frontier = PriorityQueueHeap()

        
        # min values
        # self._fmin, self._gmin, self._pmin, self._rmin = np.inf, np.inf, np.inf, np.inf

        # ================ Misc Information ===================== #
        self.closedList = {}
        self.currentF = 0
        self.currentP = 0
        self.currentNeighs = []     # Required for testing overlap using open list
        self._lmin = 0
        self.lnode = None

        # fcostsfunc is a passed-in method representing the priority, returns a float
        self.pCosts = pCostsFunc     
        self.hCosts = hCostsFunc

        #### Keep a sorted array for gmin, rmin, and fmin
        self.gmin_heap = PriorityQueueHeap()
        self.rmin_heap = PriorityQueueHeap()
        self.fmin_heap = PriorityQueueHeap()

        self.f={}
        if len(start)==1:
            # if start != "Temp":
            self.gmin_heap.put(start[0], 0)
            self.rmin_heap.put(start[0], 0)
            self.fmin_heap.put(start[0], 0)
            self.f[start[0]] = 0

            # estimate initial h costs
            # h_init = self.hCosts(self, start[0])
            # self.fmin_heap.put(start[0], h_init)
            # self.f[start[0]] = h_init

            # Extra things: not necessary for shortest path computation
            # every node will keep track its closest terminal root node based on gcost
            self.root = {start[0]: start[0]}

        # Keep track of children in shortest path computation
        # (shortest path tree rooted at a terminal)
        self.children = {}


        # to find components containing terminals
        self.findset = None

        # keep track of reference to other objs
        self.siblings = None

    # def finish_setup(self, comp_ref:dict):
    #     """Finish setting up this MultiSearch object""" 
    #     # store reference to other objs
    #     self.siblings = comp_ref

    #     if self.start != "Temp":
    #         # correct the root node f costs within the data structures
    #         self.frontier.put(self.start, self.pCosts(self, self.g, self.start))

    #         # correct rot node f costs
    #         self.fmin_heap.put(self.start, self.f[self.start])

    def finish_setup(self):
        """call this one time once all siblings have been created"""
        # finish initializing h costs
        h_init = self.hCosts(self, self.start[0])
        self.fmin_heap.put(self.start[0], h_init)
        self.f[self.start[0]] = h_init


    @property
    def goal(self):
        """Returns goals, keyed by id. The id and goal should be fixed to each other!
        
        At the moment, usage with the `Framework` class will work best
        when a dict is passed in. Else expects goal to be an iterable object,
        like a list or provide a dict

        """
        return self._goal
    
    @goal.setter
    def goal(self, goal):
        if goal is not None:
            if isinstance(goal, dict):
                self._goal = goal
            else:
                self._goal = {}
                try:
                    for ndx, k in enumerate(goal):
                        if not set((ndx,)).issubset(set(self.id)):
                            self._goal[ndx] = k
                except Exception as err:
                    my_logger.error("Issue defining goals", exc_info=True)
        else:
            self._goal = {}


    def nominate(self):
        """In this function, a node is nominated from the open set, which will later be expanded in the `update()` phase.
        
        `nominate` is done using a priority queue. The user should utilize a flag is used to 
        ensure the function is not called more than once prior to an update.

        Returns:
            True: if a node was nominated

        """
        frontier = self.frontier
        parent = self.parent
        g = self.g

        # NOTE Probably dont need this ""'nominated' ensures this function doesn't get called multiple times before update"
        if not frontier.empty():
            # current node is immediately in the closed list
            currentP, current = frontier.get_min()  # update current to be the item with best priority  
            self.current = current
            self.currentF = self.f[current]
            self.currentP = currentP

            # LOG nomination
            my_logger.debug("{} nominated {} with priority {}".format(self.id, self.current, self.currentP))

            #print("Terminal, current: ",self.start, current)
            if self.visualize:
                # Update plot with visuals
                # self.animateCurrent.update_clean(current)
                AnimateV2.add_line("nominated_{}".format(self.id), current[0], current[1], 'ko', zorder=15, draw_clean=True, markersize=10)
                # AnimateV2.update()
            # #Early exit if we reached our goal
            # if current == self.goal:
            #     return parent, g, current
            
            # return true if nominated
            return True
        
        # if no nomination
        return False

    def reprioritize(self):
        """Reprioritize the open set / frontier when heuristics change.

        For now, re-calculate each node's priority and put it into the queue.
        This is easier than searching and updating every key

        """
        # Modify frontier structure
        for o in self.frontier.entry_table.copy():
            # make sure goal is not empty
            if self.goal:
                # priority changes as a result of destination change. 
                # Hence both fmin and pmin need to be updated
                priority = self.pCosts(self, self.g, o)
                self.frontier.put(o, priority)     
                self.fmin_heap.put(o, self.f[o])

    def update(self):    
        """The open/closed list is updated here, and the open list is expanded with neighboring nodes

        For each neighbor of the nominated node (denoted as `current`), we identify its gcost,
        parent node, and priority. These 3 items are stored into 3 separate dictionaries.

        """
        frontier = self.frontier
        parent = self.parent
        g = self.g
        # current = self.current
        # frontier.delete(current)
        priority, current = frontier.get()

        # Update gmin,rmin,fmin heaps
        self.gmin_heap.delete(current)
        self.rmin_heap.delete(current)
        self.fmin_heap.delete(current)

        # self.closedList[current] = currentP
        # Delete current node from frontier
    
        #expand current node and check neighbors
        # Update stats logging
        MultiSearch.update_expanded_nodes()
        # visualize the recently closed node
        if self.visualize:             
            # self.animateClosed.update(current)

            # Delete nominated node drawing, add it as closed

            AnimateV2.add_line("closed_{}".format(self.id), current[0], current[1], 'mo', markersize=10)
            # AnimateV2.update()

            # hide the nominate node temporarily
            AnimateV2.add_line("nominated_{}".format(self.id), current[0], current[1], 'ko', alpha=0, zorder=15, draw_clean=True, markersize=10)


            # Show recently closed node with a white x (the best nominated node over all)
            # AnimateV2.add_line("recent_closed_{}".format(self.id), current[0], current[1], 'wx', alpha=1, zorder=16, draw_clean=True, markersize=10)
            # AnimateV2.update()

        # refresh neighbors
        self.currentNeighs = []

        # Add new nodes to frontier
        for next in self.graph.neighbors(current):
            g_next = g[current] + self.graph.cost(current, next)

            # if next location not in CLOSED LIST or its cost is less than before
            if next not in g or g_next < g[next]:
                # Store neighbor's gcost
                g[next] = g_next

                # update parent list
                parent[next] = current
                
                # update root node pointer (extra)
                self.root[next] = self.root[current]

                # Calculate priority and time it
                # Call priority function to get next node's priority (TODO: rename fcosts -> priority!)
                start = timer()
                priority = self.pCosts(self, g, next)
                end = timer()
                MyTimer.add_time("fcosts_time", end - start )

                # Update frontier
                frontier.put(next, priority)

                # update children before updating parent list
                if current not in self.children:
                    self.children[current] = set({next})
                else:
                    # now add children to current
                    self.children[current].add(next) 

                # make sure children of a parent is correct
                # when parents get changed
                if next in parent and parent[next] in self.children:
                    if next in self.children[parent[next]]:
                        self.children[parent[next]].remove(next)


                # update gmin,rmin, fmin heaps
                self.gmin_heap.put(next,g_next)
                self.rmin_heap.put(next, g[current])
                self.fmin_heap.put(next, self.f[next])

                # track current neighbors
                self.currentNeighs.append(next)

                
            if self.visualize:
                # self.animateNeighbors.update(next)
                # Add neighbors
                x = []
                y = []
                for n in self.frontier.elements:
                    x.append(n[0])
                    y.append(n[1])
                AnimateV2.add_line("neighbors_{}".format(self.id), x,y, 'cD', markersize=7, draw_clean=True)
                # Hide the best nominated node now
                # AnimateV2.add_line("recent_closed_{}".format(self.id), current[0], current[1], 'wx', alpha=0, draw_clean=True, markersize=10)
        
        # if self.visualize:
        #     AnimateV2.update()

        # # consider deleting fvalues to save memory, since it's only relevant to openset
        # del self.f[current]

        self.nominated = False
        # MyLogger.add_message("{} updated!".format(self.id), __name__, "DEBUG")
        my_logger.debug("{} updated!".format(self.id))


    def boundary_nodes(self):
        r = []
        for f in self.frontier.elements:
            if self.parent[f] is not None:      
                r.append(self.parent[f])
        return r

    @property
    def rmin(self):
        """Additional function to estimate min radius.
        
        Returns:
            minR (float): the minimum radius of the 'boundary nodes', i.e. closed set of nodes 
                with a child in the open set
        """
        # return minR
        if not self.rmin_heap.empty():
            value, _ = self.rmin_heap.get_test()
            return value
        else:
            return 0

        # try:
        #     value, _ = self.rmin_heap.get_test()
        #     return value
        # except Exception as e_:
        #     # return np.inf
        #     return 0
            
    @property
    def fmin(self):
        """Returns the minimum f-value from the open list

        """
        # try: 
        #     return min((self.f[k[2]] for k in self.frontier.elements))
        # except Exception as e_:
        #     # FIX: Figure out whether this should be 0 or np.inf
        #     # if open set is empty
        #     # if self.frontier.elements:
        #     #     return 0
        #     # else:
        #     #     return np.inf
        #     return self.lmin
        #     # return 0
        if not self.fmin_heap.empty():
            value, _ = self.fmin_heap.get_test()
            return value
        else:
            return 0
        # try:
        #     value, _ = self.fmin_heap.get_test()
        #     return value
        # except Exception as e_:
        #     # when frontier is empty, there is nothing else to explore!
        #     # return np.inf
        #     return 0

    @property
    def gmin(self):
        """Returns the minimum g-value from the open list

        """
        # return min((self.g[k] for k in self.frontier.elements))
        # return min(self.g[k[2]] for k in self.frontier.elements)
        if not self.gmin_heap.empty():
            value, _ = self.gmin_heap.get_test()
            return value
        else:
            return 0

        # try:
        #     value, _ = self.gmin_heap.get_test()
        #     return value
        # except Exception as e_:
        #     # return np.inf
        #     return 0

    @property
    def pmin(self):
        """Returns the minimum p-value from the open list

        """
        # return min(self.frontier.elements.values())
        try:
            priority, _ = self.frontier.get_test()
            return priority
        except:
            # return np.inf
            return 0

    @property
    def lmin(self):
        """Returns the current declared shortest path distance

        """
        return self._lmin

    @lmin.setter
    def lmin(self, val):
        """Set the current component's current shortest path distance

        """
        self._lmin = val


    def __add__(self, other):
        """Merges two `GenericSearch` objects into a single object using the '+' operator

        Merges the individual id, g list, parent list, and goals.

        Parameters:
            self (GenericSearch): Class object, left side of the '+' sign
            other (GenericSearch): Class object, right side of the '+' sign

        Example:
            mergedGS = gs1 + gs2 

        Returns:
            mergedGS (GenericSearch): class object

        Todo:
            - Refactor this method

        """       
        ## Initialize some merged structures
        mergedF = PriorityQueueHeap()    # merged frontier
        mergedG = {}                 # merged closed list/ cost_so_far
        mergedP = {}                 # merged parent list
        mergedID = []
        mergedGoal = {}
        mergedRoot = {}
        mergedChildren = {}

        ## Merge the terminal indices
        # TODO. PROB DONT NEED list
        # mergedID.extend(list(self.id))
        # mergedID.extend(list(other.id))
        mergedID.extend(self.id)
        mergedID.extend(other.id)
        # mergedID.sort()
        mergedID = tuple(mergedID)

        ## Update destinations based on indices
        # mergedGoal = {ndx: term for ndx, term in self.goal.items() if not set((ndx,)).issubset(set(mergedID))}   
        ## Make sure all components have been updated. See issue #10
        # self.update()
        # other.update()

        joint_goal_set = set(self.goal).union(set(other.goal))-set(mergedID)
        for k in joint_goal_set:
            if k in self.goal:
                mergedGoal.update({k: self.goal[k]})
            elif k in other.goal:
                mergedGoal.update({k: other.goal[k]})

        ## Create a GenericSearch Object to return
        start = []
        start.extend(self.start)
        start.extend(other.start)

        mergedGS = MultiSearch(self.graph,  start, mergedGoal, self.pCosts, visualize=cfg.Animation.visualize)

        # update id
        mergedGS.id = mergedID

        # save siblings
        mergedGS.siblings = self.siblings
        # mergedGS.finish_setup(self.siblings)

        # update set of components: delete old individuals
        self.siblings[mergedGS.id] = mergedGS
        del self.siblings[self.id]
        del self.siblings[other.id]
        
        ## new variables for ease: Linked lists, frontier, and g costs
        p1 = self.parent
        f1 = self.frontier.elements
        g1 = self.g
        c1 = set(g1) - set(f1)
        r1 = self.root

        p2 = other.parent
        f2 = other.frontier.elements
        g2 = other.g
        c2 = set(g2) - set(f2)
        r2 = other.root

        ## Get Merged g and p structures, need to handle overlapping of lists
        setG = set(g1).union(set(g2))                                 # works; handle c/o overlapping
        # closedSet = (set(g1) - set(f1)).union(set(g2) - set(f2))    # original case, working
        closedSet = c1.union(c2)
              
        # Merge the gcosts of all components (includes frontier and closed sets)
        for next in setG:
            # for overlapping nodes, retain the one with least g.
            # else, just keep them according tot he component
            if next in g1 and next in g2:
                if g1[next] < g2[next]:
                    g_next = g1[next]
                    current = p1[next]
                    root = r1[next]
                    fvalue = self.f[next]
                else:
                    g_next = g2[next]
                    current = p2[next]
                    root = r2[next]
                    fvalue = other.f[next]
            elif next in g1:
                g_next = g1[next]
                current = p1[next]
                root = r1[next]
                fvalue = self.f[next]
            elif next in g2:
                g_next = g2[next]
                current = p2[next]
                root = r2[next]
                fvalue = other.f[next]

            mergedG[next] = g_next
            mergedP[next] = current
            mergedRoot[next] = root
            mergedGS.f[next] = fvalue

            if current not in mergedChildren:
                mergedChildren[current] = set({next})
            else:
                mergedChildren[current].add(next)

        # get merged f and update merged p structures
        # setF = set(f1).union(set(f2)) - closedSet       # original case, working       
        setF = set(f1).union(set(f2))                     # works; handle c/o overlapping     

        # Recalculate the frontier costs
        # DO I NEED TO SET THE G COSTS HERE TOO?.
        # NO need to set current?
        
        # set up root
        mergedGS.root = mergedRoot
        # set up g
        mergedGS.g = mergedG
        # parent dict (linked list)
        mergedGS.parent = mergedP

        # put terminals in search.f as well
        for s in start:
            # hacky, ideally should be nonzero from beginning of algorithm
            mergedGS.f[s] = 0


        for next in setF:
            if next in f1 and next in f2:
                if g1[next] < g2[next]:
                    priority = f1[next][0]
                    current = p1[next]
                    # g_next = g1[next]
                    # Add fcosts, since they are not guaranteed to be the same as priorities
                    mergedGS.f[next] = self.f[next]
                else:
                    priority = f2[next][0]
                    current = p2[next]

                    mergedGS.f[next] = other.f[next]
                    # g_next = g2[next]
            elif next in f1:
                if next in c2 and g2[next] < g1[next]:
                    # If node is closer to terminal 2, DONT retain node in frontier of 1
                    # this was optional before
                    # mergedGS.f[next] = self.f[next]
                    continue
                elif next in c2 and g2[next] >= g1[next]:
                    # If node is closer to terminal 1, DO retain node in frontier and remove
                    # from the closed list
                    priority = f1[next][0]
                    current = p1[next]
                    # # DID I FORGET THIS?????????????????????
                    # g_next = g1[next]
                    mergedGS.f[next] = self.f[next]
                else:
                    # node doesn't overlap with c2, so retain in frontier
                    priority = f1[next][0]
                    current = p1[next]
                    # g_next = g1[next]
                    mergedGS.f[next] = self.f[next]
            elif next in f2:
                if next in c1 and g1[next] < g2[next]:
                    # this was optional before
                    # mergedGS.f[next] = self.f[next]
                    continue
                elif next in c1 and g1[next] >= g2[next]:
                    priority = f2[next][0]
                    current = p2[next]
                    mergedGS.f[next] = other.f[next]
                else:
                    priority = f2[next][0]
                    current = p2[next]
                    mergedGS.f[next] = other.f[next]


            mergedP[next] = current
            # Try updating the F costs here explicitly if mergedGoal is not empty
            # if mergedGoal:
            # ################################################
            # # REPRIORTIZING AFTER MERGE
            # ################################################
            # if cfg.Algorithm.reprioritize_after_merge:
            #     priority = self.pCosts(mergedGS, mergedG, next)
            # ################################################

            mergedF.put(next, priority)

            # Also update the gmin, rmin, fmin heaps
            mergedGS.gmin_heap.put(next, mergedG[next])
            if current is None:
                mergedGS.rmin_heap.put(next, 0)
            else:
                mergedGS.rmin_heap.put(next, mergedG[current])
            mergedGS.fmin_heap.put(next, mergedGS.f[next])

            # mergedP[next] = current

        ################################################
        # REPRIORTIZING AFTER MERGE
        ################################################
        if cfg.Algorithm.reprioritize_after_merge:
            for next in setF:
                if next in c2 and g2[next] < g1[next] or \
                    next in c1 and g1[next] < g2[next]:
                    # If node is closer to terminal 2, DONT retain node in frontier of 1
                    # this was optional before
                    # mergedGS.f[next] = self.f[next]
                    continue
                # Try updating the F costs here explicitly if mergedGoal is not empty
                # if mergedGoal:
                priority = self.pCosts(mergedGS, mergedG, next)
                ################################################
                mergedGS.fmin_heap.put(next, mergedGS.f[next])
                mergedF.put(next, priority)

        # # removed start="Temp" from frontier and related heaps
        # mergedGS.frontier.delete('Temp')
        # mergedGS.fmin_heap.delete("Temp")
        # mergedGS.gmin_heap.delete("Temp")
        # mergedGS.rmin_heap.delete("Temp")


        # set closed list, valued by currentF
    
        # Set current node and currentF    
        # if self.currentF < other.currentF:
        #     mergedGS.currentF = self.currentF
        #     mergedGS.current = self.current
        # else:
        #     mergedGS.currentF = other.currentF
        #     mergedGS.current = other.current

        ## modify generic search object values
        # mergedGS.g = mergedG
        # mergedGS.parent = mergedP
        mergedGS.frontier = mergedF
        mergedGS.children = mergedChildren
        # if g1[self.current] < g2[other.current]
        # if self.currentF < other.currentF:
        #     mergedGS.current = self.current
        #     mergedGS.currentF = self.currentF
        # else:
        #     mergedGS.current = other.current
        #     mergedGS.currentF = other.currentF

        # mergedGS.nominated = True

        # TODO also initialize closed List..but you really dont need to
        # mergedGS.closedList =

        # Set lmin? NOTE don't!!!!
        # mergedGS.lmin = min(self.lmin, other.lmin)
        # mergedGS.lmin = max(self.lmin, other.lmin)

        ## Update plot colors
        if cfg.Animation.visualize:
            # mergedGS.animateClosed.order=10
            # mergedGS.animateClosed.update(np.array(list(closedSet)).T.tolist())      # remember to pass a structure of size 2
            # mergedGS.animateNeighbors.update(np.array(list(setF)).T.tolist())      # remember to pass a structure of size 2

            # Delete previous drawings
            AnimateV2.delete("nominated_{}".format(self.id))
            AnimateV2.delete("closed_{}".format(self.id))
            AnimateV2.delete("neighbors_{}".format(self.id))            
            AnimateV2.delete("nominated_{}".format(other.id))
            AnimateV2.delete("closed_{}".format(other.id))
            AnimateV2.delete("neighbors_{}".format(other.id))

            # Draw new merged components
            dataClosedSet = np.array(list(closedSet)).T.tolist()
            dataSetF = np.array(list(setF)).T.tolist()
            AnimateV2.add_line("closed_{}".format(mergedGS.id), dataClosedSet[0], dataClosedSet[1], 'mo', markersize=10)
            # if for some reason the open set is empty, don't draw
            if dataSetF:
                AnimateV2.add_line("neighbors_{}".format(mergedGS.id), dataSetF[0], dataSetF[1], 'cD', markersize=7, draw_clean=True)
            else:
                AnimateV2.add_line("neighbors_{}".format(mergedGS.id), [], [], 'cD', markersize=7, draw_clean=True)

        return mergedGS
        
        


            
