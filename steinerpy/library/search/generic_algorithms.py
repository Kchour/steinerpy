"""This module provides a generic incremental search class, that breaks up nomination and update phase"""



import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer

import steinerpy.config as cfg
from steinerpy.library.animation import AnimateV2
from steinerpy.library.logger import MyLogger
from steinerpy.library.misc.utils import MyTimer
from steinerpy.library.search.search_utils import PriorityQueue, PriorityQueueHeap
from steinerpy.library.search.search_utils import DoublyLinkedList

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
        fCostsFunc: A function returning the fCosts of node u. Returns a scalar value. Arguments are (self, glist, next_node)
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
    # total_closed_nodes = 0
    # total_opened_nodes = 0
    total_expanded_nodes = 0
    def __init__(self, graph, start, goal, frontierType, fCostsFunc, id):
        # ================ Required Definitions ===================== #
        self.graph = graph                         
        self.start = start
        self.goal = goal     
        self.current = None

        # Open List
        self.frontier = frontierType                
        self.frontier.put(start, 0)
        
        # The cost so far (includes both frontier and closed list)
        # TODO: May have to modify g merging in the future for primal-dual
        self.g = {}
        self.g[start] = 0

        # Linked List
        self.parent = {}
        self.parent[start] = None

        # F costs function object for priority updates
        self.fCosts = fCostsFunc     # fCostsFunc is a passed-in method, returns a float

        # root node
        self.root = {start: start}

    # def set_start(self, start):
    #     self.start = start

    ################################################
    ###   Class methods for updating some stats  ###
    ################################################

    @classmethod
    def update_expanded_nodes(cls):
        cls.total_expanded_nodes +=1
    
    # @classmethod
    # def update_opened_nodes(cls):
    #     cls.total_opened_nodes +=1

    # @classmethod
    # def update_closed_nodes(cls):
    #     cls.total_closed_nodes +=1

    @classmethod
    def reset(cls):
        """Reset all class variables """
        cls.total_expanded_nodes = 0

    def reconstruct_path(self, parents, goal, start=None, order='forward'):
        '''Given a linked list, rebuild a path back from a goal node 
            to a start (or root node, if start is not specified)

        paremeters:
            parents: a singly-linked list using python dict
            start: a tuple (x,y) position. Optional
            goal: a tuple (x,y) position. Mandatory
            order: 'forward', or 'reverse' 

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
            current = parents.get(current, None)

        if start != None:
            path.append(start)
        if order == 'forward':
            path.reverse()

        return path

class GenericSearch(Search):
    """This class extends `Search` and breaks up the search into `nominate` and `update` phases

    This gives the user finer control over the search space, i.e. when to stop, update destinations midway, etc.
    `GenericSearch` also inherits all the attributes of `Search`

    Attributes:
        visualize (bool): A flag for visualizing the algorithm. Mainly for debug purposes
        animateCurrent (Animate): Animate the current nominated node
        animateClosed (Animate): Animate the history of the closed set
        animateNeighbors (Animate): Animate the open list in the `update`

    Todo:
        * Consider putting animateClosed in the `update` function, because closing does not occur until `update`
    """
    def __init__(self, graph,  fCostsFunc, start, frontierType, goal=None, visualize=False, id=None):
        Search.__init__(self, graph, start, goal, frontierType, fCostsFunc, id)
        
        # Visualize algorithm flag
        self.visualize = visualize

        # Keep track of nomination status
        self.nominated = False       # Make sure we don't nominate twice in a row

        # Each search object needs an id
        self.id = (id,)

        # keep track of F
        self.f = {}
        self.f[start] = 0            #May need to figure out how to initialize this besides 0
        
        # min values
        # self._fmin, self._gmin, self._pmin, self._rmin = np.inf, np.inf, np.inf, np.inf

        # ================ Misc Information ===================== #
        self.closedList = {}
        self.currentF = 0
        self.currentP = 0
        self.currentNeighs = []     # Required for testing overlap using open list
        self._lmin = 0
        self.lnode = None

        #### Keep a sorted array for gmin, rmin, and fmin
        self.gmin_heap = PriorityQueueHeap()
        self.rmin_heap = PriorityQueueHeap()
        self.fmin_heap = PriorityQueueHeap()

        self.gmin_heap.put(start, 0)
        self.rmin_heap.put(start, 0)
        self.fmin_heap.put(start, 0)

        # Visulization?
        # if visualize:     
        #     # initialize plot (graph has the dimensions built it)
        #     # xlim = (graph.grid_dim[0], graph.grid_dim[1])   #(minX, maxX)
        #     # ylim = (graph.grid_dim[2], graph.grid_dim[3])   #(minY, maxY)
        #     # no sleep atm
        #     # self.animateCurrent = Animate(number=1, xlim=xlim, ylim=ylim, gridSize=1,linewidth=5, markerType='bo', markerSize=10, sleep=0, order=2)
        #     # self.animateClosed = Animate(number=1, xlim=xlim, ylim=ylim, gridSize=1,linewidth=5, markerType='o', markerSize=10, sleep=0, order=-1)
        #     # self.animateNeighbors = Animate(number=1, xlim=xlim, ylim=ylim, gridSize=1,linewidth=5, markerType='o', markerSize=5, sleep=0, order=-1)
        #     # self.animatePath = Animate(number=1, xlim=xlim, ylim=ylim, gridSize=1,linewidth=5, markerType='o', markerSize=5, sleep=0.000, order=-1)
        #     pass

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
        if isinstance(goal, dict):
            self._goal = goal
        else:
            self._goal = {}
            try:
                for ndx, k in enumerate(goal):
                    if not set((ndx,)).issubset(set(self.id)):
                        self._goal[ndx] = k
            except Exception as err:
                print(err)


    def nominate(self):
        """In this function, a node is nominated from the open set, which essentially updates the open set.
        
        `nominate` is done using a priority queue. A flag is used in the conditional to 
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
            currentP, current = frontier.get_test()  # update current to be the item with best priority  
            self.current = current
            self.currentF = self.f[current]
            self.currentP = currentP

            # LOG nomination
            MyLogger.add_message("{} nominated {} with priority {}".format(self.id, self.current, self.currentP), __name__, "DEBUG")

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
                priority = self.fCosts(self, self.g, o)
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
        GenericSearch.update_expanded_nodes()
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
                
                # Calculate priority and time it
                # Call priority function to get next node's priority (TODO: rename fcosts -> priority!)
                start = timer()
                priority = self.fCosts(self, g, next)
                end = timer()
                MyTimer.add_time("fcosts_time", end - start )

                # Update frontier and parent list
                frontier.put(next, priority)
                parent[next] = current

                # update root node pointer
                self.root[next] = self.root[current]

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




        # consider deleting fvalues to save memory, since it's only relevant to openset
        del self.f[current]

        self.nominated = False
        MyLogger.add_message("{} updated!".format(self.id), __name__, "DEBUG")


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
        # minR = None
        # for f in self.frontier.elements:
        #     # when starting off the parent is none
        #     if self.parent[f] is None:
        #         minR = 0
        #     else:
        #         # check the gcost of boundary nodes
        #         r = self.g[self.parent[f]]
        #         if minR is None or r < minR:
        #             minR = r

        # if minR is None:
        #     minR = 0

        # return minR
        try:
            value, _ = self.rmin_heap.get_test()
            return value
        except Exception as e_:
            return np.inf
            
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
        try:
            value, _ = self.fmin_heap.get_test()
            return value
        except Exception as e_:
            # when frontier is empty, there is nothing else to explore!
            return np.inf

    @property
    def gmin(self):
        """Returns the minimum g-value from the open list

        """
        # return min((self.g[k] for k in self.frontier.elements))
        # return min(self.g[k[2]] for k in self.frontier.elements)
        try:
            value, _ = self.gmin_heap.get_test()
            return value
        except Exception as e_:
            return np.inf

    @property
    def pmin(self):
        """Returns the minimum p-value from the open list

        """
        # return min(self.frontier.elements.values())
        try:
            priority, _ = self.frontier.get_test()
            return priority
        except:
            return np.inf

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
        mergedF = PriorityQueueHeap()    # merged frontier   #tricky, priorityQueue or priorityQueueHeap?
        mergedG = {}                 # merged closed list/ cost_so_far
        mergedP = {}                 # merged parent list
        mergedID = []
        mergedGoal = {}
        mergedRoot = {}

        ## Merge the terminal indices
        # TODO. PROB DONT NEED list
        # mergedID.extend(list(self.id))
        # mergedID.extend(list(other.id))
        mergedID.extend(self.id)
        mergedID.extend(other.id)
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
        mergedGS = GenericSearch(self.graph,  self.fCosts, 'Temp', mergedF, goal=mergedGoal, visualize=cfg.Animation.visualize)
        
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
                else:
                    g_next = g2[next]
                    current = p2[next]
                    root = r2[next]
            elif next in g1:
                g_next = g1[next]
                current = p1[next]
                root = r1[next]
            elif next in g2:
                g_next = g2[next]
                current = p2[next]
                root = r2[next]

            mergedG[next] = g_next
            mergedP[next] = current
            mergedRoot[next] = root

        # get merged f and update merged p structures
        # setF = set(f1).union(set(f2)) - closedSet       # original case, working       
        setF = set(f1).union(set(f2))                     # works; handle c/o overlapping     

        # Recalculate the frontier costs
        # DO I NEED TO SET THE G COSTS HERE TOO?.
        # NO need to set current?
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
                    continue
                elif next in c1 and g1[next] >= g2[next]:
                    priority = f2[next][0]
                    current = p2[next]
                    mergedGS.f[next] = other.f[next]
                else:
                    priority = f2[next][0]
                    current = p2[next]
                    mergedGS.f[next] = other.f[next]


            # Try updating the F costs here explicitly if mergedGoal is not empty
            # if mergedGoal:
            ################ COMMENT AS NEEDED ##############
            priority = self.fCosts(mergedGS, mergedG, next)

            mergedF.put(next, priority)
            # Also update the gmin, rmin, fmin heaps
            mergedGS.gmin_heap.put(next, mergedG[next])
            if current is None:
                mergedGS.rmin_heap.put(next, 0)
            else:
                mergedGS.rmin_heap.put(next, mergedG[current])
            mergedGS.fmin_heap.put(next, mergedGS.f[next])

            mergedP[next] = current
            # mergedG[next] = g_next

        # removed start="Temp" from frontier and related heaps
        mergedGS.frontier.delete('Temp')
        mergedGS.fmin_heap.delete("Temp")
        mergedGS.gmin_heap.delete("Temp")
        mergedGS.rmin_heap.delete("Temp")


        # set closed list, valued by currentF
    
        # Set current node and currentF    
        # if self.currentF < other.currentF:
        #     mergedGS.currentF = self.currentF
        #     mergedGS.current = self.current
        # else:
        #     mergedGS.currentF = other.currentF
        #     mergedGS.current = other.current

        ## modify generic search object values
        mergedGS.g = mergedG
        mergedGS.parent = mergedP
        mergedGS.id = mergedID
        mergedGS.frontier = mergedF
        mergedGS.root = mergedRoot
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
            AnimateV2.add_line("neighbors_{}".format(mergedGS.id), dataSetF[0], dataSetF[1], 'cD', markersize=7, draw_clean=True)

        return mergedGS
        
        


            
