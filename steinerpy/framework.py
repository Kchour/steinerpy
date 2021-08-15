"""This module defines a 'Framework' class which is to be inherited by S* family of algorithms"""


import numpy as np
from timeit import default_timer as timer
from typing import List
from abc import ABC, abstractmethod
import cProfile
import logging, logging.config

from steinerpy.library.misc.abc_utils import abstract_attribute, ABC as newABC
from steinerpy.library.graphs.graph import IGraph
# from steinerpy.library.animation.animation import Animate, SaveAnimation
from steinerpy.library.animation import AnimateV2
from steinerpy.library.search.search_utils import PriorityQueue, reconstruct_path, PriorityQueueHeap
from steinerpy.library.search.generic_algorithms import GenericSearch
import steinerpy.config as cfg
from .abstract_algo import AbstractAlgorithm
from .algorithms.common import Common
# from steinerpy.library.logger import MyLogger #deprecated
from steinerpy.library.misc.utils import MyTimer

# configure and create logger
my_logger = logging.getLogger(__name__)

class Framework(AbstractAlgorithm):
    """This class serves as a foundation for S*.

    Any algorithm can be adapted to use this framework. Override member functions as needed
    
    Attributes:
        terminals (list): A list of tuples representing terminals on a graph. 
            Exact format depends on the type of graph used (see below).
            Inherited from `AbstractAlgorithm`
        graph (SquareGrid, MyGraph): Graph classes from superclass IGraph.
            Created using 'GraphFactory' class from the 'graph' module.
            Inhereited from `AbstractAlgorithm`
        FLAG_STATUS_pathConverged (bool): Initially `False`, used to check for path convergence
        FLAG_STATUS_completeTree (bool): Initially `False`, used to end any algorithm
        comps (dict): A table of search components, which can be merged
        F (dict): A working forest or cache to store closed nodes. Used to check for intersections.
        nodeQueue (PriorityQueue): Used to pop nominated nodes
        pathQueue (PriorityQueueHeap): Used to pop candidate 'shortest paths'
        run_debug (int): A debugging counter, incremented per iteration
        selNode (tuple): The popped node from the nomination queue above.
        selData (dict): Information pertaining to the selected node `selNode`.
    
    Other:
        Attributes:
            animateS (Animate): Used to animate the solution. Mainly for debugging

    The `run_algorithm` function is the crux of this class, which performs the following,        
    (all algorithms inheriting this class, must be adapted to use the this function)

    Example:
        >>> while(not self.FLAG_STATUS_completeTree):
        >>>    self.nominate()
        >>>    self.update()
        >>>    self.path_check()
        >>>    self.tree_update()
        >>>    self.tree_check()
        
    Todo: 
        * Ensure `PriorityQueueHeap` respects order during tie-breaking
        * Include logging features and run levels
        * Include custom logger for data collection
        * Don't do `if array`, to check whether it's empty
        * Add support for generic graphs

    """

    def __init__(self,G: IGraph, T: List):
        AbstractAlgorithm.__init__(self, G,T)
        # self.terminals = T
        # self.graph = G
        # self.S = {'sol':[], 'dist':[], 'path':[]}

        # RUNTIME OK AND STATUS FLAGS 
        # self.FLAG_OK_NOMINATE_OK = False
        # self.FLAG_OK_UPDATE_OK = False
        # self.FLAG_OK_PCHECK_OK = False
        # self.FLAG_OK_TCHECK_OK = False 
        self.FLAG_STATUS_pathConverged = False
        self.FLAG_STATUS_completeTree = False


        # Create search algorithm objects
        self.comps = Common.create_search_objects(search_class=GenericSearch, 
                                                graph=self.graph, 
                                                f_costs_func=self.f_costs_func,
                                                frontier_type=PriorityQueueHeap, 
                                                terminals=self.terminals, 
                                                visualize=cfg.Animation.visualize
                                                )

        self.findset = {}

        # self.comps = {(index, ):  GenericSearch(self.graph, self.f_costs_func, start, frontierType=PriorityQueue(), \
        #             goal={i: self.terminals[i] for i in set(range(len(self.terminals)))-set((index,))}, visualize=cfg.visualize, id=index) \
        #             for index,start in enumerate(self.terminals) }

        # Create cache and solution set
        self.F = {}
        self.nodeQueue = PriorityQueueHeap()
        self.pathQueue = PriorityQueueHeap()
        # Keep track of number of iterations
        self.run_debug = 0
        # recently closed node and its data
        self.selNode = None
        self.selData = None
        # Whether we are using depots
        if "Depot" in str(type(G)):
            self.use_depots = True
            self.depots = G.depots
        else:
            self.use_depots = False
            self.depots = None


        self.testone = []
        self.testIntersection = []

        # UFeasPath...
        # An adjacency list of each existing component. Each entry keyed by {comp2: [feasible path dist, common node]}
        # if there exist a feasible path between comp1 and comp2, then the key-entry will exist in the adj list of comp1
        # Every iteration of the algorithm, we try to update [feas. path dist, comm node] with a lower valued dist
        # Also during merge, we have to update the keys in both the UFeasPath AND the subkeys...

        self.UFeasPath = {}
        # cycle detection Necessary for Astar
        # self.cd = CycleDetection([(t,) for t in range(len(terminals))])

        # TODO: FIX THIS UP, plot related
        if cfg.Animation.visualize:
            minX, maxX, minY, maxY = self.graph.grid_dim
            # grid_size = self.graph.grid_size
            # # plot terminals
            # self.plotTerminals = Animate(number=1, xlim=(minX, maxX), ylim=(minY, maxY), gridSize=grid_size, linewidth=5, markerType='ko', markerSize=20, sleep=0.000, order=1)
            # self.plotTerminals.update(np.array(self.terminals).T.tolist())

            # # plot update current
            # self.plotCurrent = Animate(number=1, xlim=(minX, maxX), ylim=(minY, maxY), gridSize=grid_size, linewidth=5, markerType='wo', markerSize=10, sleep=0.000, order=5)

            # # plot solution
            # self.animateS = Animate(number=1, xlim=(minX, maxX), ylim=(minY, maxY), gridSize=grid_size, linewidth=5, markerType='ro', markerSize=10, sleep=0.000, order=50)

            # # if any obstacles
            # if self.graph.obstacles:
            #     self.plotObstacle = Animate(number=1, xlim=(minX, maxX), ylim=(minY, maxY), gridSize=grid_size, linewidth=5, markerType='o', markerSize=2, sleep=0.5, order=50)
            #     self.plotObstacle.update(np.array(self.graph.obstacles).T.tolist())

        
            if self.run_debug <= 1:

                # Add obstacles
                if self.graph.obstacles:
                    # AnimateV2.add_line("obstacles", np.array(self.graph.obstacles).T.tolist(), markersize=5, marker='o', color='k')
                    # fig, ax = plt.subplots()
                        
                    fig, ax = AnimateV2.create_new_plot(num=1)
                    #ax.matshow(grid_data, cmap='seismic')
                    im_artist = ax.imshow(self.graph.grid, cmap='Greys', origin='lower')

                    AnimateV2.init_figure(fig, ax, xlim=(minX, maxX), ylim=(minY,maxY))

                    AnimateV2.add_artist_ex(im_artist, "obstacles")
    
                # Add terminal points using add_line function
                AnimateV2.add_line("terminals", np.array(self.terminals).T.tolist(), xlim=(minX, maxX), ylim=(minY,maxY), markersize=15, mew=5, marker="x", color='red',zorder=11)

                if self.depots is not None:
                    AnimateV2.add_line("depots", np.array(self.depots).T.tolist(), markersize=14, marker='o', color='b', zorder=12)

                # # Add stuff to figure 2
                # AnimateV2.add("terminals",  np.array(self.terminals).T.tolist(), figure_number=2,xlim=(minX, maxX), ylim=(minY,maxY), markersize=15, marker="x", color='r',zorder=11)
                # if self.graph.obstacles:
                #     AnimateV2.add("obstacles", np.array(self.graph.obstacles).T.tolist(), figure_number=2, markersize=5, marker='o', color='k')
                
                # if self.depots is not None:
                #     AnimateV2.add("depots", np.array(self.depots).T.tolist(), figure_number=2, markersize=14, marker='o', color='b')

                # AnimateV2.update(figure_number=2)    
                AnimateV2.update()       

    def nominate(self):
        """Each component nominates a node from its open set
               
        For each component in `comps`, we nominate a single node. 
        If the nominate function returns a `True`, we move on to
        storing the nominated node and its priority value (Fcost)
        in a queue variable called nodeQueue.
        
        """
        # print(self.run_debug)
        # MyLogger.add_message("performing nominate() ", __name__, "INFO")
        my_logger.info("performing nomination")

        for ndx, c in self.comps.items():
            if ndx not in self.nodeQueue.elements:
                if c.nominate():
                    self.nodeQueue.put(ndx, c.currentP)

    def update(self):
        """ Update the component's open/closed list pertaining to the least fcost-valued node
        
        Among all nominated nodes, choose the one with least f cost using a priority queue.

        """
        # MyLogger.add_message("performing update() ", __name__, "INFO")
        my_logger.info("performing update")
        # CONSIDER USING TRY CATCH FOR THIS ENTIRE LOOP

        try:
            # TEST 
            #self.nodeQueue = PriorityQueue()
            # nodeQueue is empty!
            if self.nodeQueue.empty():
                print(self.terminals)
                raise Exception("nodeQueue is empty!")

            best_priority, best_ndx = self.nodeQueue.get()  
        except Exception as e:
            # MyLogger.add_message("nodeQueue has an error", __name__, "ERROR", "exc_info=True")
            my_logger.error("nodeQueue has an error", exc_info=True)
            raise e

        # Get best ndx from priority queue
        
        # get best component object, and g cost of best node
        bestC = self.comps[best_ndx]
        bestCurrent = bestC.current
        bestGVal = bestC.g[bestCurrent]  
        bestPVal = bestC.currentP
        
        # Get parent (if possible), parent is a dict
        bestParent = bestC.parent.get(bestCurrent, None)

        # Store and return the selected node(s)
        self.selNode = bestCurrent
        self.selData = {t:{} for t in self.terminals} 
        self.selData.update({'to': bestParent, 'terminalInd': best_ndx, 'gcost': bestGVal, 'pcost':bestPVal, 'status': 'closed'})

        # print('selected Node: ', self.selNode)
        # print('selected Data: ', self.selData)
        # Check for complete path

        # Now update the closed/open list for the 'best' component. Make sure non-empty goals
        # if self.comps[best_ndx].goal:
        # hold off on updating?
        bestC.update()
        # self.comps[self.selData['terminalInd']].update()
        # if cfg.visualize:
        #     self.plotCurrent.update_clean(self.selNode)

        #clear node queue?
        # self.nodeQueue = PriorityQueue()


    def path_check(self):
        """Check for and add paths if able, to the `pathQueue`

        More specifically:
            1. Check for all set collisions
            2. Find shortest path among two colliding sets
            3. If path convergence criteria is satisfied, add to pathQueue

        TEST purposes: 
            * Try adding all nodes with finite g-cost value

        """
        # newNodes = {}
        # for k in self.comps[self.selData['terminalInd']].currentNeighs:
        #     comp = self.comps[self.selData['terminalInd']]
        #     bestParent = comp.parent[k]
        #     best_ndx = comp.id
        #     gcost = comp.g[k]
        #     pcost = comp.frontier.elements[k]

        #     newNodes[k] = {'to': bestParent, 'terminalInd': best_ndx, 'gcost': gcost, 'pcost': pcost, 'status':'open'}

        # # add recently closed node from previous function
        # newNodes[self.selNode] = self.selData

        t1 = self.selData['terminalInd']
        updatedComp = self.comps[t1]
        for t2, c in self.comps.items():
            # skip if trying to intersect with itself
            if t1 == t2:
                continue
        
            # duplicates = False
            # for p in self.pathQueue.elements.values():
            #     terms_ind, terms_actual, path, dist = p[2] 
            #     if set((t1,t2)).issubset(set(terms_ind)):
            #         duplicates = True
            #         break
            # if duplicates:
            #     continue
            # Avoid adding duplicate paths to the path PriorityQueue
            if (t1,t2) in self.pathQueue.elements or (t2,t1) in self.pathQueue.elements:
                continue

            # Add updated component recent open and closed list
            updateSet = set(updatedComp.currentNeighs)
            # updateSet = set(updatedComp.frontier.elements)
            updateSet.add(updatedComp.current)
            # if c.currentNeighs:
            #     updateSet.union(set(c.currentNeighs))
            # updateSet.add(c.current)

            # # look at previous results
            if t1 in self.UFeasPath and t2 in self.UFeasPath[t1]:
                # UFeas, commonNode = self.UFeasPath[(t1,t2)][0], self.UFeasPath[(t1,t2)][1]
                if self.UFeasPath[t1][t2][0] < self.UFeasPath[t2][t1][0]:
                    UFeas, commonNode = self.UFeasPath[t1][t2][0], self.UFeasPath[t1][t2][1]
                    self.UFeasPath[t2][t1] = [UFeas, commonNode]
                else:
                    UFeas, commonNode = self.UFeasPath[t2][t1][0], self.UFeasPath[t2][t1][1]
                    self.UFeasPath[t1][t2] = [UFeas, commonNode]
                # # ensure symmetry
                # if t2 in self.UFeasPath:
                #     self.UFeasPath[t2].update({t1: [UFeas, commonNode]})
                # else:                           
                #     self.UFeasPath.update({t2: {t1: [UFeas, commonNode]}})                          
            else:
                UFeas = None
         
            for k in updateSet:
                if k in self.comps[t1].g and k in self.comps[t2].g:
                    candU = self.comps[t1].g[k] + self.comps[t2].g[k]
                    if  UFeas is None or candU < UFeas:
                        UFeas = candU
                        commonNode = k
                        # if (t1,t2) in self.UFeasPath:
                        #     self.UFeasPath[(t1,t2)] = [candU, k]
                        # elif (t2, t1) in self.UFeasPath:
                        #     self.UFeasPath[(t2,t1)] = [candU, k]
                        # else:
                        #     self.UFeasPath[(t1,t2)] = [candU, k]
                        
                        if t1 in self.UFeasPath:
                            self.UFeasPath[t1].update({t2: [UFeas, commonNode]})
                        else:
                            self.UFeasPath.update({t1: {t2: [UFeas, commonNode]}})
                        
                        if t2 in self.UFeasPath:
                            self.UFeasPath[t2].update({t1: [UFeas, commonNode]})
                        else:                           
                            self.UFeasPath.update({t2: {t1: [UFeas, commonNode]}})  



            # UFeas = None                
            # # updateSet = set(updatedComp.frontier.elements)
            # for k in updateSet:
            #     if k in self.comps[t1].g and k in self.comps[t2].g:
            #         candU = self.comps[t1].g[k] + self.comps[t2].g[k]
            #         if  UFeas is None or candU < UFeas:
            #             UFeas = candU
            #             commonNode = k

            ######################################################################################    
            # try:
            #     jointSet = ((c.g[k] + updatedComp.g[k], k) for k in set(c.g).intersection(set(updatedComp.frontier.elements)))
            #     minItem = min(jointSet)
            #     # UFeas = minItem[0]
            #     # commonNode = minItem[1]
            #     del jointSet
            # except:
            #     # UFeas, commonNode = None, None
            #     pass
            ######################################################################################
            # if abs(UFeas - 11.656) < 1e-3:
            #     self.testone.append(UFeas)
            if UFeas is not None:
                # set lmins for each component
                if UFeas < updatedComp.lmin or updatedComp.lmin == 0:
                    updatedComp.lmin = UFeas
                if UFeas < c.lmin or c.lmin == 0:
                    c.lmin = UFeas

                # Subtract some slack due to numerical issues
                # t1, t2 = t1feas, t2feas
                sp = self.shortest_path_check(comps=self.comps, term_edge=(t1,t2), bestVal=UFeas)
                if sp:
                    # try:
                    #     jointSet = ((c.g[k] + updatedComp.g[k], k) for k in set(c.g).intersection(set(updatedComp.g)))
                    #     minItem = min(jointSet)
                    #     UFeas = minItem[0]
                    #     commonNode = minItem[1]
                    #     del jointSet
                    # except:
                    #     # UFeas, commonNode = None, None
                    #     pass

                    # if criteria is satisfied, update the path queue               
                    # Get path based on sel_node



                    # path, dist, term_actual = Common.get_path(comps=self.comps, sel_node=commonNode, term_edge=(t1,t2),\
                    #     reconstruct_path_func = reconstruct_path)




                    ###########################################
                    ### # update destination list TEST THIS ###
                    ###########################################
                    # MyLogger.add_message("goals(PRE) of {} is {}".format(t1, self.comps[t1].goal), __name__, "Debug")
                    # MyLogger.add_message("goals(PRE) of {} is {}".format(t2, self.comps[t2].goal), __name__, "Debug")
                    try:
                        # # Consider not deleting me from you
                        pass
                        # test =1
                        for t in t2:
                            if t in self.comps[t1].goal:
                                del self.comps[t1].goal[t]
                        
                        for t in t1:
                            if t in self.comps[t2].goal:
                                del self.comps[t2].goal[t]

                    except Exception as e_:
                        print(e_)
                        print("")
                        # MyLogger.add_message("Update goal error!", __name__, "ERROR", "exc_info=True")
                        my_logger.error("Update goal error!", exc_info=True)
                        raise e_

                    # # reprioritze
                    if self.comps[t1].goal:
                        self.comps[t1].reprioritize()

                    if self.comps[t2].goal:
                        self.comps[t2].reprioritize()  

                    # # Delete respective components from nodeQueue
                    if t1 in self.nodeQueue.elements:
                        self.nodeQueue.delete(t1)
                    if t2 in self.nodeQueue.elements:
                        self.nodeQueue.delete(t2)

                    # MyLogger.add_message("goals(POST) of {} is {}".format(t1, self.comps[t1].goal), __name__, "Debug")
                    # MyLogger.add_message("goals(POST) of {} is {}".format(t2, self.comps[t2].goal), __name__, "Debug")

                    ############################################
                    # ## End update destination list and rep ###
                    # ##########################################  

                    # MyLogger.add_message("paths in solution set: {}".format(len(self.S['dist'])), __name__, "INFO")
                    my_logger.info("paths in solution set: {}".format(len(self.S['dist'])))

                    # # Set another lower bound on components due to declared shortest path
                    # if dist > 0 and dist < self.comps[t1].lmin or self.comps[t1].lmin == 0:
                    #     self.comps[t1].lmin = dist

                    # if dist > 0 and dist < self.comps[t2].lmin or self.comps[t2].lmin == 0:
                    #     self.comps[t2].lmin = dist

                    # when component frontier is empty, the most recently declared path is fmin
                    # self.comps[t1].lmin = dist
                    # self.comps[t2].lmin = dist            
        
                    # self.pathQueue.put(({'terms': (t1,t2), 'term_actual': term_actual, 'path':path, 'dist':dist}), dist)
                    # self.pathQueue.put((( (t1,t2), term_actual, tuple(path), dist)), dist)
                    self.pathQueue.put((t1,t2), UFeas)

                    # MyLogger.add_message("Added path to pathQueue", __name__, "DEBUG")

                    # MyLogger.add_message("pathQueue len now: {}".format(len(self.pathQueue.elements)), __name__, "INFO")

                    my_logger.debug("Added path to pathQueue")
                    my_logger.info("pathQueue len now: {}".format(len(self.pathQueue.elements)))


                    if cfg.Misc.DEBUG_MODE:
                        self.debug_fmin()
                        self.debug_gmin
                        self.debug_pmin()
                        # self.debug_lmin()
                        self.debug_rmin()
                        testtesttest=1
        # MyLogger.add_message("performing path_check() ", __name__, "INFO")

        # ### We check for set collisions using a cache F
        # # Add neighbors
        # newNodes = {}
        # for k in self.comps[self.selData['terminalInd']].currentNeighs:
        #     comp = self.comps[self.selData['terminalInd']]
        #     bestParent = comp.parent[k]
        #     best_ndx = comp.id
        #     gcost = comp.g[k]
        #     pcost = comp.frontier.elements[k]

        #     newNodes[k] = {'to': bestParent, 'terminalInd': best_ndx, 'gcost': gcost, 'pcost': pcost, 'status':'open'}

        # # add recently closed node from previous function
        # newNodes[self.selNode] = self.selData

        # # check updated component's newly added nodes
        # for ndx, nodeInfo in enumerate(newNodes.items()):
        #     node, data = nodeInfo
        #     # Node to be looked at
        #     MyLogger.add_message("Looking at node {}".format(node), __name__, "Debug")

        #     # check for meet collision
        #     self.iscollided = Common.set_collision_check(sel_node=node, sel_data=data,\
        #         target_list=self.F, cache=self.F )

        #     # verify meet criteria
        #     if self.iscollided:
        #         # Define terminal indices
        #         t1 = data['terminalInd']            
        #         t2 = self.F[node]['terminalInd']    

        #         # Update cache with lower valued gcost
        #         if data['gcost'] < self.F[node]['gcost']:
        #             self.F[node] = data

        #         if t1 == t2:
        #             continue

        #         # Check for duplicates, but allow at least 1 duplicate if node is closed in both directions
        #         duplicates = False
        #         for p in self.pathQueue.elements:
        #             if set((t1,t2)).issubset(set(p[2]['terms'])):
        #                 duplicates = True
        #                 break
                
        #         # Check for shortest paths (i.e nicholsons or similar)
        #         sp = False
        #         if not duplicates:
        #             # Get shortest path and perform criteria check
        #             bestVal, bestNode, bestDict = Common.get_shortest_path(comps=self.comps, term_edge=(t1, t2),\
        #                 f_costs_func=self.f_costs_func)
        #             sp = self.shortest_path_check(comps=self.comps, term_edge=(t1,t2), bestVal=bestVal)
                    
        #             ################### TEST
        #             # self.t1test, self.t2test = Common.subsetFinder(((5,),(33,)), self.comps)

        #             # if abs(bestVal - 1.414) <1e-3:
        #             #                     # Get path based on sel_node
        #             #     path, dist, term_actual = Common.get_path(comps=self.comps, sel_node=bestNode, term_edge=(t1,t2),\
        #             #         reconstruct_path_func = reconstruct_path)
        #             #     # if 5 not in self.comps[self.t1test].goal and 33 not in self.comps[self.t2test].goal and \
        #             #     #     5 not in self.comps[self.t2test].goal and 33 not in self.comps[self.t1test].goal:
        #             #     #     print("destination changed?")
        #             #     self.testone += 1
        #                 # self.tester = 1
        #             ####################
        #             # Set another lower bound on components due to declared shortest path
        #             # if bestVal > 0 and bestVal < self.comps[t1].lmin or self.comps[t1].lmin == 0:
        #             #     self.comps[t1].lmin = bestVal

        #             # if bestVal > 0 and bestVal < self.comps[t2].lmin or self.comps[t2].lmin == 0:
        #             #     self.comps[t2].lmin = bestVal

        #             if cfg.console_level == "DEBUG":
        #                 MyLogger.add_message("Collision between {},{}".format(t1, t2), __name__, "Debug")
        #                 MyLogger.add_message("t1 {} has terminals {}".format(t1, [self.terminals[k] for k in t1]), __name__, "Debug")
        #                 MyLogger.add_message("t2 {} has terminals {}".format(t2, [self.terminals[k] for k in t2]), __name__, "Debug")
        #                 MyLogger.add_message("Feasible Path dist {}".format(bestVal), __name__, "Debug")
        #                 MyLogger.add_message("sel node n {}".format(node), __name__, "Debug")
        #                 MyLogger.add_message("{} gcosts(n) {}".format(t1, self.comps[t1].g[node] ), __name__, "Debug")
        #                 MyLogger.add_message("{} gcosts(n) {}".format(t2, self.comps[t2].g[node]), __name__, "Debug")
        #                 MyLogger.add_message("{} current fmin {}".format(t1, self.comps[t1].fmin), __name__, "Debug")
        #                 MyLogger.add_message("{} current fmin {}".format(t2, self.comps[t2].fmin), __name__, "Debug")
        #                 MyLogger.add_message("{} current gmin {}".format(t1, self.comps[t1].gmin), __name__, "Debug")
        #                 MyLogger.add_message("{} current gmin {}".format(t2, self.comps[t2].gmin), __name__, "Debug")
        #                 MyLogger.add_message("{} current pmin {}".format(t1, self.comps[t1].pmin), __name__, "Debug")
        #                 MyLogger.add_message("{} current pmin {}".format(t2, self.comps[t2].pmin), __name__, "Debug")
                    
        #             if sp: 
        #                 MyLogger.add_message("Feasible path TO BE ADDED with dist {}".format(bestVal), __name__, "DEBUG")
        #             else:
        #                 MyLogger.add_message("Feasible path NOT ADDED with dist {}".format(bestVal), __name__, "DEBUG")

        #             # update
        #             # self.selNode, self.selData = bestNode, bestDict
                
        #     # if criteria is satisfied, update the path queue
        #     if self.iscollided and sp and not duplicates:      
                
        #         # Get path based on sel_node
        #         path, dist, term_actual = Common.get_path(comps=self.comps, sel_node=bestNode, term_edge=(t1,t2),\
        #             reconstruct_path_func = reconstruct_path)

        #         ###########################################
        #         ### # update destination list TEST THIS ###
        #         ###########################################
        #         MyLogger.add_message("goals(PRE) of {} is {}".format(t1, self.comps[t1].goal), __name__, "Debug")
        #         MyLogger.add_message("goals(PRE) of {} is {}".format(t2, self.comps[t2].goal), __name__, "Debug")
        #         try:
        #             # # Consider not deleting me from you
        #             pass
        #             # test =1
        #             for t in t2:
        #                 if t in self.comps[t1].goal:
        #                     del self.comps[t1].goal[t]
                    
        #             for t in t1:
        #                 if t in self.comps[t2].goal:
        #                     del self.comps[t2].goal[t]
        #             # # test = len(set(self.comps[t1].goal).union(set(self.comps[t2].goal)))
        #             # del self.comps[t1].goal[t2[0]]
        #             # del self.comps[t2].goal[t1[0]]
        #             # united_goals = set(self.comps[t2].goal).intersection(set(self.comps[t1].goal))
        #             # # # # # # if self.comps[t1].goal:
        #             # self.comps[t1].goal = {i:self.terminals[i] for i in united_goals}
        #             # # # # # # if self.comps[t2].goal:
        #             # self.comps[t2].goal = {i:self.terminals[i] for i in united_goals}
        #             # self.comps[self.selData['terminalInd']].update()

        #             # for t in map(lambda x: self.terminals.index(x),term_actual):
        #             #     if t in self.comps[t1].goal:
        #             #         del self.comps[t1].goal[t]
        #                 # if t in self.comps[t2].goal:
        #                 #     del self.comps[t2].goal[t]

        #         except Exception as e_:
        #             print(e_)
        #             print("")
        #             MyLogger.add_message("Update goal error!", __name__, "ERROR", "exc_info=True")
        #             raise

        #         # reprioritze
        #         if self.comps[t1].goal:
        #             self.comps[t1].reprioritize()

        #         if self.comps[t2].goal:
        #             self.comps[t2].reprioritize()  

        #         # # Delete respective components from nodeQueue
        #         self.nodeQueue.delete(t1)
        #         self.nodeQueue.delete(t2)

        #         MyLogger.add_message("goals(POST) of {} is {}".format(t1, self.comps[t1].goal), __name__, "Debug")
        #         MyLogger.add_message("goals(POST) of {} is {}".format(t2, self.comps[t2].goal), __name__, "Debug")

        #         ############################################
        #         # ## End update destination list and rep ###
        #         # ##########################################  

        #         MyLogger.add_message("paths in solution set: {}".format(len(self.S['dist'])), __name__, "INFO")

        #         # # Set another lower bound on components due to declared shortest path
        #         # if dist > 0 and dist < self.comps[t1].lmin or self.comps[t1].lmin == 0:
        #         #     self.comps[t1].lmin = dist

        #         # if dist > 0 and dist < self.comps[t2].lmin or self.comps[t2].lmin == 0:
        #         #     self.comps[t2].lmin = dist

        #         self.comps[t1].lmin = dist
        #         self.comps[t2].lmin = dist            
    
        #         self.pathQueue.put({'terms': (t1,t2), 'term_actual': term_actual, 'path':path, 'selData':bestDict, 'selNode': bestNode, 'dist':dist}, dist)
        #         MyLogger.add_message("Added path to pathQueue", __name__, "DEBUG")

        #         MyLogger.add_message("pathQueue len now: {}".format(len(self.pathQueue.elements)), __name__, "INFO")

        #         if cfg.console_level == "DEBUG":
        #             self.debug_fmin()
        #             self.debug_gmin
        #             self.debug_pmin()
        #             self.debug_lmin()
        #             self.debug_rmin()
        #             testtesttest=1
        # # self.comps[self.selData['terminalInd']].update()


    def tree_update(self):
        """ Empty the pathQueue if possible, then update the solution set `S`

        For each possible path, we perform a `merge` function on the components connected by this path.
        The resulting merged component is stored in `comps` and the former components are deleted.
        As soon as a solution set is updated, we allow for a `tree_check` in the future.

        """
        my_logger.info("performing tree_update")

        # Empty path queue, gather up the solutions in solution queue (FIFO)
        solQueue = Common.solution_handler(comps=self.comps, path_queue=self.pathQueue, cycle_detector=None, \
            terminals=self.terminals, criteria=self.path_queue_criteria, merging=True, use_depots=self.use_depots)
        
        my_logger.info("solQueue len: {}".format(len(solQueue)))

        # add paths to solution set
        for ndx, s in enumerate(solQueue):
            # self.add_solution(s['path'], s['dist'], s['terms'])
            # t1,t2 = s['terms']
            my_logger.debug("emptying solQueue iter: {}".format(ndx+1))

            # t1, t2 = Common.subsetFinder(s['terms'], self.comps)
            # MyLogger.add_message("Inspecting path {}. Old Comps {}. New Comps {}. Terminals {}. length {}".format(s['path'], s['terms'], (t1,t2), s['term_actual'], s['dist']), __name__, "DEBUG")

            t1,t2 = s['components']     #could be old components
            # To avoid adding redundant paths. 


            pdist = s['dist']
            
            # find set t1 and t2 belong to
            if t1[0] in self.findset: 
                t1 = self.findset[t1[0]]
            if t2[0] in self.findset:
                t2 = self.findset[t2[0]]
            
            if t1 == t2:
                continue

            # debug bounds
            # self.debug_bounds(t1)
            # self.debug_bounds(t2)
        
            # update findset 
            for t in (t1+t2):
                self.findset[t] = (t1+t2)
              
            # Get common node between two components
            dist, commonNode = self.UFeasPath[t1][t2]

            if abs(pdist-dist)>0.1:
                # print("")
                raise ValueError("distances don't match! path queue and feasible table is conflicting!", self.terminals, self, pdist, dist)

            # reconstruct path
            path, _, term_actual = Common.get_path(comps=self.comps, sel_node=commonNode, term_edge=(t1,t2),\
            reconstruct_path_func = reconstruct_path)

            try:
                
                # MyLogger.add_message("Just added path {}. Terminals {}. length {}".format(s['path'], s['term_actual'], s['dist']), __name__, "INFO")

                Common.add_solution(path=path, dist=dist, edge=term_actual,\
                    solution_set=self.S, terminals=self.terminals)

                # True as soon as we add a solution
                self.FLAG_STATUS_pathConverged = True

                # self.comps[t1].lmin = s['dist']
                # self.comps[t2].lmin = s['dist']

                # # update stuff in the solQueue
                # for sol in solQueue:
                    

                # # Update path Queue element keys to refer to updated component indices
                # for c in self.comps:
                #     if (t1, c) in self.pathQueue.entry_table:
                #         self.pathQueue.put((t1+t2,c), self.pathQueue.entry_table[(t1,c)][0])
                #         self.pathQueue.delete((t1, c))
                #     if (c, t1) in self.pathQueue.entry_table:
                #         self.pathQueue.put((c,t1+t2), self.pathQueue.entry_table[(c,t1)][0])
                #         self.pathQueue.delete((c, t1))
                #     if (t2, c) in self.pathQueue.entry_table:
                #         self.pathQueue.put((t1+t2,c), self.pathQueue.entry_table[(t2,c)][0])
                #         self.pathQueue.delete((t2, c))
                #     if (c, t2) in self.pathQueue.entry_table:
                #         self.pathQueue.put((c,t1+t2), self.pathQueue.entry_table[(c,t2)][0])
                #         self.pathQueue.delete((c, t2))

                # Perform merging
                # Common.merge_comps(self.comps, term_edge=(t1, t2), nodeQueue=self.nodeQueue, cache=self.F)
                Common.merge_comps(self.comps, term_edge=(t1,t2), nodeQueue=self.nodeQueue, cache=self.F)

                # Update feasible path keys and subkeys to refer to updated component indices                
                # minimize over exclusive union
                set1 = set(self.UFeasPath[t1])              # components adjacent to compt1
                set2 = set(self.UFeasPath[t2])              # components adjaceny to compt2
                ex_union = set1.union(set2)-set({t1,t2})    # all adjacent components excluding comp1,comp2
                merged_key = {t1+t2: {}}
                delList = []
                for k in ex_union:
                    # Create adjacency list for the newly merged component:
                    # For components adjacent to both comp1 and comp2 indvidually, store the shorter feasible path
                    # elif for components adjacent to only comp1 or comp2, just take the feas path directly
                    if k in set1 and k in set2:
                        if self.UFeasPath[t1][k][0] < self.UFeasPath[t2][k][0]:
                            merged_key[t1+t2].update({k: [self.UFeasPath[t1][k][0], self.UFeasPath[t1][k][1]]})      
                        else:
                            merged_key[t1+t2].update({k: [self.UFeasPath[t2][k][0], self.UFeasPath[t2][k][1]]})       
                    elif k in set1:
                        merged_key[t1+t2].update({k: self.UFeasPath[t1][k]})
                    elif k in set2:
                        merged_key[t1+t2].update({k: self.UFeasPath[t2][k]})

                    # update old sub-keys to point to merged comp
                    # if kth-component was adjacent to t1 or t2, update its adj list
                    if t1 in self.UFeasPath[k] and t2 in self.UFeasPath[k]:
                        # make sure to take minimum! Dont blindly set this
                        if self.UFeasPath[k][t1] < self.UFeasPath[k][t2]:
                            self.UFeasPath[k].update({t1+t2: self.UFeasPath[k][t1]})
                        else:
                            self.UFeasPath[k].update({t1+t2: self.UFeasPath[k][t2]})
                        delList.append((k,t1))
                        delList.append((k,t2))
                    elif t1 in self.UFeasPath[k]:
                        self.UFeasPath[k].update({t1+t2: self.UFeasPath[k][t1]})
                        delList.append((k,t1))
                        # del self.UFeasPath[k][t1]
                    elif t2 in self.UFeasPath[k]:
                        self.UFeasPath[k].update({t1+t2: self.UFeasPath[k][t2]})
                        delList.append((k,t2))
                        # del self.UFeasPath[k][t2]

                # delete old unmerged comps
                del self.UFeasPath[t1]
                del self.UFeasPath[t2]

                # delete old sub-keys
                for d in delList:
                    del self.UFeasPath[d[0]][d[1]]
                
                # Add merged comp
                self.UFeasPath.update(merged_key)

                # Log f costs after merging
                # MyLogger.add_message("{} current fmin {}".format(t1+t2, self.comps[t1+t2].fmin), __name__, "Debug")
    
                # TODO find a better way to animate path
                if cfg.Animation.visualize:
                #     self.animateS.update_clean(np.vstack(self.S['path']).T.tolist())

                #     self.plotTerminals.update(np.array(self.terminals).T.tolist())
                #     if self.graph.obstacles:
                #         self.plotObstacle.update(np.array(self.graph.obstacles).T.tolist())

                    AnimateV2.add_line("solution", np.vstack(self.S['path']).T.tolist(), 'yo', markersize=10, zorder=10)
                    # # if self.graph.obstacles:
                    pass

            except Exception as e:
                my_logger.error("Merging error!", exc_info=True)
                print(self.terminals)
                raise e

            my_logger.info("Total tree edges now: {}".format(len(self.S['dist'])))
                   
        my_logger.info("pathQueue len now: {}".format(len(self.pathQueue.elements)))

    def tree_check(self):
        """When at least one solution has been added to the solution set, check to see if we have a complete tree"""

        my_logger.info("Performing tree_check")

        if cfg.Animation.visualize:
            # Don't plot every thing for large graphs
            if np.mod(self.run_debug, np.ceil(self.graph.edge_count()/5000))==0:
                AnimateV2.update()

        if self.FLAG_STATUS_pathConverged:
           
            # Check tree size
            if len(self.S['sol']) == len(self.terminals)-1:
                # Algorithm has finished
                self.FLAG_STATUS_completeTree = True
                totalLen = sum(np.array(self.S['dist']))

                my_logger.info("Finished: {}".format(totalLen))

                # Add expanded node stats
                self.S['stats']['expanded_nodes'] = GenericSearch.total_expanded_nodes
                # Reset or "close" Class variables

                # Add additional stats (don't forget to reset classes)
                self.S['stats']['fcosts_time'] = sum(MyTimer.timeTable["fcosts_time"])

                # Keep plot opened
                if cfg.Animation.visualize:
                    # ### Redraw closed + frontier regions with fixed color
                    # # k = self.comps.keys()
                    # # self.comps[k[0]].g
                    # # self.comps[k[0]]
                    # # xo = []
                    # # yo = []
                    # # for n in self.comps[k[0]].frontier.elements:
                    # #     xo.append(n[0])
                    # #     yo.append(n[1])
                    # # AnimateV2.add("closed_{}".format(self.comps[k[0]].id), dataClosedSet[0], dataClosedSet[1], 'o', markersize=10, draw_clean=True)
                    # # AnimateV2.add("neighbors_{}".format(self.comps[k[0]].id), xo, yo, 'D', color='c', markersize=10, draw_clean=True)
                    
                    # recolor the final plot so that all open sets, closed sets have fixed color
                    terminal_handle = None
                    for artist, art_dict in AnimateV2.instances[1].artists.items():
                        # Set closed sets to the same color: magenta
                        if "closed" in artist:
                            art_dict['artist'][0].set_markerfacecolor("magenta")
                            art_dict['artist'][0].set_markerfacecoloralt("magenta")
                            art_dict['artist'][0].set_markeredgecolor("magenta")

                        # Set open sets to the same color: cyan
                        if "neighbors" in artist:
                            art_dict['artist'][0].set_markerfacecolor("cyan")
                            art_dict['artist'][0].set_markerfacecoloralt("cyan")
                            art_dict['artist'][0].set_markeredgecolor("cyan")

                        # Set open sets to the same color: cyan
                        if "solution" in artist:
                            art_dict['artist'][0].set_markerfacecolor("yellow")
                            art_dict['artist'][0].set_markerfacecoloralt("yellow")
                            art_dict['artist'][0].set_markeredgecolor("yellow")


                        if "terminal" in artist:
                            terminal_handle = art_dict['artist'][0]
                    AnimateV2.update()

                    import matplotlib.pyplot as plt
                    from matplotlib.patches import Rectangle
                    closed_rect = Rectangle((0, 0), 0, 0 , fc="magenta", fill=True, edgecolor=None, angle=45, linewidth=1)
                    open_rect = Rectangle((0, 0), 0, 0 , fc="cyan", fill=True, edgecolor=None, linewidth=1, angle=45)
                    sol_rect = Rectangle((0, 0), 0, 0 , fc="yellow", fill=True, edgecolor=None, linewidth=1)

                    labels = ["closed-set", "open-set", "tree-path", 'terminals']

                    plt.legend([closed_rect, open_rect, sol_rect, terminal_handle], labels, ncol=4, bbox_to_anchor=(0, 1.10, 1, 0), loc="lower left")
                    # import re
                    # search = re.search(r'.*algorithms.(\w+).(\w+)', str(self))
                    # alg_name = search.group(2)
                    # ax = AnimateV2.instances[1].ax
                    # ax.set_title(alg_name)

                    plt.tight_layout()  #to make legend fit
                    ax = plt.gca()
                    ax.axis('equal')
                    # ax.set_aspect('equal', 'box')

                    plt.draw()
                    plt.pause(1)

            self.FLAG_STATUS_pathConverged = False

    def debug_nongoals(self):
        my_logger.debug("SHOWING EXCLUDED GOALS")
        my_logger.debug("GGGGGGGGGGGGGGGGGGGGGG")
        for c in self.comps:
            c_terms= [self.terminals[t] for t in c]
            nongoals = set(range(len(self.terminals))) - set(self.comps[c].goal)
            nongoal_terms = [self.terminals[t] for t in nongoals]
            # MyLogger.add_message("comp: {}, not in goal: {}, comp_terms: {}, goal_terms: {}".format(c, nongoals, c_terms, nongoal_terms), __name__, "DEBUG")
            my_logger.debug("comp: {}, not in goal: {}, comp_terms: {}, goal_terms: {}".format(c, nongoals, c_terms, nongoal_terms))
    def debug_bounds(self, c):
        print("comp: ", c, "fmin: ", self.comps[c].fmin, "gmin: ", self.comps[c].gmin, "lmin: ", self.comps[c].lmin)

    def debug_fmin(self):
        # Log f-values
        my_logger.debug("SHOWING FMIN GOALS")
        my_logger.debug("FFFFFFFFFFFFFFFFFFF")
        minVal = None
        for i,j in self.comps.items(): 
            # MyLogger.add_message("comp: {}, terminals: {}, fmin: {}".format(i, [self.terminals[k] for k in i], j.fmin), __name__, "DEBUG")
            my_logger.debug("comp: {}, terminals: {}, fmin: {}".format(i, [self.terminals[k] for k in i], j.fmin))
            if minVal is None or j.fmin < minVal:
                minVal= j.fmin
        return minVal

    def debug_gmin(self):
        # Log g-values
        my_logger.debug("SHOWING GMIN VALUES")
        my_logger.debug("GGGGGGGGGGGGGGGGGGG")
        minVal = None
        for i,j in self.comps.items(): 
            # MyLogger.add_message("comp: {}, terminals: {}, gmin: {}".format(i, [self.terminals[k] for k in i], j.gmin), __name__, "DEBUG")
            my_logger.debug("comp: {}, terminals: {}, gmin: {}".format(i, [self.terminals[k] for k in i], j.gmin))
            if minVal is None or j.gmin < minVal:
                minVal= j.gmin
        return minVal

    def debug_pmin(self):
        # Log p-values
        my_logger.debug("SHOWING PMIN VALUES")
        my_logger.debug("PPPPPPPPPPPPPPPPPPP")
        minVal = None
        for i,j in self.comps.items(): 
            # MyLogger.add_message("comp: {}, terminals: {}, pmin: {}".format(i, [self.terminals[k] for k in i], j.pmin), __name__, "DEBUG")
            my_logger.debug("comp: {}, terminals: {}, pmin: {}".format(i, [self.terminals[k] for k in i], j.pmin))
            if minVal is None or j.pmin < minVal:
                minVal= j.pmin
        return minVal

    def debug_lmin(self):
        my_logger.debug("SHOWING LMIN VALUES")
        my_logger.debug("LLLLLLLLLLLLLLLLLLL")
        minVal = None
        for i,j in self.comps.items(): 
            # MyLogger.add_message("comp: {}, terminals: {}, lmin: {}".format(i, [self.terminals[k] for k in i], j.lmin), __name__, "DEBUG")
            my_logger.debug("comp: {}, terminals: {}, lmin: {}".format(i, [self.terminals[k] for k in i], j.lmin))
            if minVal is None or j.lmin < minVal:
                minVal= j.lmin
        return minVal

    def debug_rmin(self):
        my_logger.debug("SHOWING RMIN VALUES")
        my_logger.debug("RRRRRRRRRRRRRRRRRRR")
        minVal = None
        for i,j in self.comps.items(): 
            # MyLogger.add_message("comp: {}, terminals: {}, rmin: {}".format(i, [self.terminals[k] for k in i], j.rmin), __name__, "DEBUG")
            my_logger.debug("comp: {}, terminals: {}, rmin: {}".format(i, [self.terminals[k] for k in i], j.rmin))
            if minVal is None or j.rmin < minVal:
                minVal = j.rmin
        return minVal    

    @abstractmethod
    def f_costs_func(self):
        """An abstract method used to calculate the fcost of a node in the open set. Must be overriden! 
        
        Typically, this is the sum of g and h, i.e. f=g+h, but may depend on the user's case
        
        """
        pass 

    @abstractmethod
    def h_costs_func(self):
        """An abstract method used to calculate the heuristic cost of a node in the open set. Must be overriden! 
        
        See the `Common` class, which has a staticmethod for typical grid-based heuristics
        
        """
        pass

    def shortest_path_check(self, **kwargs):
        """Applies a necessary condition to check whether two colliding components possess the `shortest possible path`
        between them. Speficailly, the necessary condition is based on Nicholson's Bidirectional Search paper.

        Just because the two components collide (i.e. their closed sets touch), does not mean any complete path
        between them is necessarily the shortest. Override this method as needed for customized check, otherwise,
        Nicolson's from `Common` class is used.
        
        """

        return Common.shortest_path_check(**kwargs)

    def path_queue_criteria(self, **kwargs):
        """Used to pop the shortest paths from `pathQueue`, by ensuring path's cost does not exceed the other path estimates.

        Ensures steiner tree edges are added in the order of increasing cost. Override this method as needed for a different check.

        """
        return Common.path_queue_criteria(**kwargs)

    ''' 'where the magic happens' function '''
    def run_algorithm(self):
        """Query the algorithm to generate results in the solution set `S`.

        The algorithm will terminate when there is a minimum-spanning tree in S.

        Returns:
            True: if algorithm has terminated successfully
        
        Resource:
            https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python
        
        """

        # fmin, rmin, lmin, gmin
        # self.bound_tests = [[],[],[],[]]
        if cfg.Misc.profile_frame:
            cpr = cProfile.Profile()
            cpr.enable()  

        #start time
        startLoop = timer()
        while(not self.FLAG_STATUS_completeTree):

            # self.bound_tests[0].append(self.debug_fmin())
            # self.bound_tests[1].append(self.debug_rmin())
            # self.bound_tests[2].append(self.debug_lmin())
            # self.bound_tests[3].append(self.debug_gmin())

            my_logger.debug("============================================================")
            my_logger.debug("Start of Loop: {}".format(self.run_debug))

            # self.fmin_test.append(self.debug_fmin())
            # if (49, 34, 5) in self.comps:
            #     # 33 not in goal
            #     print("STOP")  

            # if (83,33) in self.comps:
            #     # 5 not in goal
            #     print("STOP")  
            start = timer()
            self.nominate()
            end = timer()
            MyTimer.add_time("nominate()_time", end - start )

            start = timer()
            self.update()
            end = timer()
            MyTimer.add_time("update()_time", end - start )

            start = timer()
            self.path_check()
            end = timer()
            MyTimer.add_time("path_check()_time", end - start )

            start = timer()
            self.tree_update()
            end = timer()
            MyTimer.add_time("tree_update()_time", end - start )

            start = timer()
            self.tree_check()
            end = timer()
            MyTimer.add_time("tree_check()_time", end - start )
            
            # keep track of loops (debugging purposes)
            my_logger.debug("End of Loop: {}".format(self.run_debug))
            self.run_debug += 1


        # # TEST PLOT FMIN_DEBUG
        # AnimateV2.add("fmin_debug", [x for x in range(len(self.fmin_test))], self.fmin_test, xlim=(0,self.run_debug), markersize=14, marker="o", color='r',zorder=11)
        # AnimateV2.update()
        # # plt.show()
        # import matplotlib.pyplot as plt

        # End time, store in stats
        endLoop = timer()
        self.S['stats']['time'] = endLoop - startLoop
        self.S['stats']['iterations'] = self.run_debug

        if cfg.Misc.profile_frame:
            cpr.disable()
            cpr.print_stats(sort='time')

        # DEBUG
        # for a in zip(self.bound_tests[0], self.bound_tests[1], self.bound_tests[2], self.bound_tests[3]): print(a)

        # Store each functions time taken (see above)
        listOfFuncs = ["nominate()_time", "update()_time", "path_check()_time",\
        "tree_update()_time", "tree_check()_time" ]
        for n in listOfFuncs:
            self.S['stats'][n] = sum(MyTimer.timeTable[n])

        # Reset classes (find a better way to to do this)
        GenericSearch.reset()
        MyTimer.reset()

        # Sucessfully generated a tree     
        return True
        # try:
    
        #     return True
        # except Exception as _e:
        #     return False


    