"""This module defines a 'Framework' class which is to be inherited by S* family of algorithms"""


import numpy as np
from timeit import default_timer as timer
from typing import List
from abc import ABC, abstractmethod
import cProfile
import logging, logging.config
import matplotlib.pyplot as plt
from decimal import Decimal

from steinerpy.library.misc.abc_utils import abstract_attribute, ABC as newABC
from steinerpy.library.graphs.graph import IGraph
from steinerpy.library.animation import AnimateV2
from steinerpy.library.search.search_utils import PriorityQueueHeap
from steinerpy.library.search.search_algorithms import MultiSearch
import steinerpy.config as cfg
from .abstract_algo import AbstractAlgorithm
from .common import Common
from .heuristics import Heuristics
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
        FLAG_STATUS_COMPLETE_TREE (bool): Initially `False`, used to end any algorithm
        comps (dict): A table of search components, which can be merged
        F (dict): A working forest or cache to store closed nodes. Used to check for intersections.
        node_queue (PriorityQueue): Used to pop nominated nodes
        path_queue (PriorityQueueHeap): Used to pop candidate 'shortest paths'
        run_debug (int): A debugging counter, incremented per iteration
        selNode (tuple): The popped node from the nomination queue above.
        selData (dict): Information pertaining to the selected node `selNode`.
    
    Other:
        Attributes:
            animateS (Animate): Used to animate the solution. Mainly for debugging

    The `run_algorithm` function is the crux of this class, which performs the following,        
    (all algorithms inheriting this class, must be adapted to use the this function)

    Example:
        >>> while(not self.FLAG_STATUS_COMPLETE_TREE):
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
        # self.results = {'sol':[], 'dist':[], 'path':[]}

        # RUNTIME OK AND STATUS FLAGS 
        # self.FLAG_OK_NOMINATE_OK = False
        # self.FLAG_OK_UPDATE_OK = False
        # self.FLAG_OK_PCHECK_OK = False
        # self.FLAG_OK_TCHECK_OK = False 
        self.FLAG_STATUS_PATH_CONVERGED = False
        self.FLAG_STATUS_COMPLETE_TREE = False

        # Create search algorithm objects
        self.comps = Common.create_search_objects(search_class=MultiSearch, 
                                                graph=self.graph, 
                                                p_costs_func=self.p_costs_func,
                                                h_costs_func=self.h_costs_func,
                                                terminals=self.terminals, 
                                                visualize=cfg.Animation.visualize
                                                )

        # to find component containing said terminal 
        # self.findset = {t: c for t,c in zip(self.terminals, self.comps)}
        # share reference to comp dict 
        for c in self.comps.values():
            c.siblings = self.comps
            # c.findset = self.findset
            # c.finish_setup(self.comps)
        
        # now finish setting up
        for c in self.comps.values():
            c.finish_setup()
            

        # make sure root nodes 
        self.findset = {}


        # Create cache and solution set
        self.node_queue = PriorityQueueHeap()
        self.path_queue = PriorityQueueHeap()
        self.global_bound_queue = PriorityQueueHeap()

        # keep track of shortest paths added already...
        self.sol_edges = set()

        # Keep track of number of iterations
        self.run_debug = 0

        # Whether we are using depots
        if "Depot" in str(type(G)):
            self.use_depots = True
            self.depots = G.depots
        else:
            self.use_depots = False
            self.depots = None

        # UFeasPath...
        # An adjacency list of each existing component. Each entry keyed by {comp2: [feasible path dist, common node]}
        # if there exist a feasible path between comp1 and comp2, then the key-entry will exist in the adj list of comp1
        # Every iteration of the algorithm, we try to update [feas. path dist, comm node] with a lower valued dist
        # Also during merge, we have to update the keys in both the UFeasPath AND the subkeys...
        self.UFeasPath = {}

        # Plotting Related
        # TODO: Make this more efficient
        # TODO: Use mayaVI for 3d grids
        if cfg.Animation.visualize:        
            if self.run_debug <= 1:
                # if not plt.fignum_exists(1):
                #     # if figure doesn't exist yet, create it
                #     fig, ax = AnimateV2.create_new_plot(num=1, figsize=(7,7))
                # else:
                #     # get ax and fig if they exist
                #     ax = plt.gca()
                #     fig = plt.gcf()
                fig, ax = AnimateV2.create_new_plot(figsize=(7,7))

                if cfg.Algorithm.graph_domain == "grid":
                    # get dimensions of grid
                    minX, maxX, minY, maxY = self.graph.grid_dim
                    AnimateV2.init_figure(fig, ax, xlim=(minX, maxX), ylim=(minY,maxY))
                    # ax.autoscale()

                     # Add obstacles
                    if self.graph.obstacles:
                        # AnimateV2.add_line("obstacles", np.array(self.graph.obstacles).T.tolist(), markersize=5, marker='o', color='k')
                        AnimateV2.add_line("obstacles", self.graph.obstacles(), markersize=1, marker='.', color='k')
                            
                        #ax.matshow(grid_data, cmap='seismic')
                        im_artist = ax.imshow(self.graph.grid, cmap='Greys', origin='lower', alpha=0)

                        AnimateV2.add_artist_ex(im_artist, "obstacles")
               
                elif cfg.Algorithm.graph_domain == "generic":

                    ax.autoscale()
                    AnimateV2.init_figure(fig,ax)
                    AnimateV2.update()
                
                # Add terminal points ("terminals")
                data = np.array(self.terminals)
                terminal_artist = ax.plot(data[:,0], data[:,1], markersize=15, mew=5, marker="x", linestyle="", color='red',zorder=9)    
    
                # we can only call add_artist_ex() after init_figure() is called
                # keep track of terminal artist
                # also plot will return an array[artist]
                # whereas scatter does not
                AnimateV2.add_artist_ex(terminal_artist[0], "terminal")

                if self.depots is not None:
                    data = np.array(self.depots)
                    blah = ax.plot(data[:,0], data[:,1], markersize=14, linestyle="", marker='o', color='b', zorder=12)
                    AnimateV2.add_artist_ex(blah[0], "depots")

                # render to screen
                AnimateV2.update()       

    def nominate(self):
        """Each component nominates a node from its open set
               
        For each component in `comps`, we nominate a single node. 
        If the nominate function returns a `True`, we move on to
        storing the nominated node and its priority value (Fcost)
        in a queue variable called node_queue.
        
        """
        my_logger.info("performing nomination")

        for ndx, c in self.comps.items():
            if ndx not in self.node_queue or cfg.Algorithm.always_nominate:
                if cfg.Algorithm.reprioritize_before_nominations:
                    c.reprioritize()

                if c.nominate():
                    self.node_queue.put(ndx, c.currentP)
                    # also update global bound
                    self.update_global_bound(ndx)

        # Ensure each component has a nomination (this fails if we run out of nodes to explore)
        assert(len(self.node_queue)==len(self.comps))

    def update(self):
        """ Update the component's open/closed list pertaining to the least fcost-valued node
        
        Among all nominated nodes, choose the one with least f cost using a priority queue.

        """
        # MyLogger.add_message("performing update() ", __name__, "INFO")
        my_logger.info("performing update")

        try:
            # This exception can only happen when our graph is disconnected
            if self.node_queue.empty():
                print(self.terminals)
                raise Exception("node_queue is empty!")

            best_priority, best_ndx = self.node_queue.get()  
        except Exception as e:
            # MyLogger.add_message("node_queue has an error", __name__, "ERROR", "exc_info=True")
            my_logger.error("node_queue has an error", exc_info=True)
            raise e

        # update the least cost component
        self.comps[best_ndx].update()

        # store this index
        self.best_ndx = best_ndx

        # keep track of global bounds
        # Need to modify IDs during merge
        self.update_global_bound(best_ndx)

    def path_check(self):
        """Check for and add paths if able, to the `path_queue`

        More specifically:
            1. Check for all set collisions
            2. Find shortest path among two colliding sets
            3. If path convergence criteria is satisfied, add to path_queue

        TEST purposes: 
            * Try adding all nodes with finite g-cost value

        """

        c1 = self.best_ndx
        for c2 in self.comps.keys():
            
            # skip if trying to intersect with itself
            if c1 == c2:
                continue

            # track updated component's current and neighboring node. Need to check for intersection
            update_set = set(self.comps[c1].currentNeighs)
            update_set.add(self.comps[c1].current)
       

            # Avoid adding duplicate paths to the path PriorityQueue
            # if (c1,c2) in self.path_queue or (c2,c1) in self.path_queue or \
            #     (c1, c2) in self.sol_edges or (c2, c1) in self.sol_edges:
            #     continue

            # Store only the best possible feasible path so far
            # Obtain previous feasible path result
            if c1 in self.UFeasPath and c2 in self.UFeasPath[c1]:
                if self.UFeasPath[c1][c2][0] < self.UFeasPath[c2][c1][0]:
                    self.UFeasPath[c2][c1] = self.UFeasPath[c1][c2]
                else:
                    self.UFeasPath[c1][c2] = self.UFeasPath[c2][c1]                        
                UFeas, commonNode = self.UFeasPath[c1][c2]
            else:
                UFeas = None
         
            # try to discover new better paths between (c1, c2)
            for k in update_set:
                if k in self.comps[c1].g and k in self.comps[c2].g:
                    candU = self.comps[c1].g[k] + self.comps[c2].g[k]
                    if  UFeas is None or candU < UFeas:
                        UFeas = candU
                        commonNode = k  
                        
                        if c1 in self.UFeasPath:
                            self.UFeasPath[c1].update({c2: [UFeas, commonNode]})
                        else:
                            self.UFeasPath.update({c1: {c2: [UFeas, commonNode]}})
                        
                        if c2 in self.UFeasPath:
                            self.UFeasPath[c2].update({c1: [UFeas, commonNode]})
                        else:                           
                            self.UFeasPath.update({c2: {c1: [UFeas, commonNode]}})  

            if UFeas is not None:
                # Set lmins for each component. May be important post-merge
                if UFeas < self.comps[c1].lmin or self.comps[c1].lmin == 0:
                    self.comps[c1].lmin = UFeas
                    self.comps[c1].lnode = commonNode
                if UFeas < self.comps[c2].lmin or self.comps[c2].lmin == 0:
                    self.comps[c2].lmin = UFeas
                    self.comps[c2].lnode = commonNode

                my_logger.debug("ITERATION {}".format(self.run_debug)) 
                my_logger.debug("Observing edge between {} {} - cost {}, local fmin1 {} fmin2 {}, gmin1 {} gmin2 {} global LB {}".\
                    format(c1,c2, UFeas, self.comps[c1].fmin, self.comps[c2].fmin, self.comps[c1].gmin, self.comps[c2].gmin, self.global_bound_queue.get_min()))

                sp = self.shortest_path_check([c1,c2], UFeas)

                if cfg.Misc.DEBUG_MODE:
                    self.debug_fmin()
                    self.debug_gmin()
                    self.debug_pmin()
                    # self.debug_lmin()
                    self.debug_rmin()

                if sp:
                    my_logger.debug("Adding sp edge between {} {} - cost {}, local fmin1 {} fmin2 {}, gmin1 {} gmin2 {}".\
                        format(c1,c2, UFeas, self.comps[c1].fmin, self.comps[c2].fmin, self.comps[c1].gmin, self.comps[c2].gmin))

                    ###########################################
                    ### # update destination list TEST THIS ###
                    ###########################################
                    # MyLogger.add_message("goals(PRE) of {} is {}".format(t1, self.comps[t1].goal), __name__, "Debug")
                    # MyLogger.add_message("goals(PRE) of {} is {}".format(t2, self.comps[t2].goal), __name__, "Debug")
                    try:
                        # don't delete goals if only 2 components left!
                        # sometimes extra expansions are necessary before 
                        # the merge works
                        if len(self.comps)>2:
                            for t in c2:
                                if t in self.comps[c1].goal:
                                    del self.comps[c1].goal[t]
                            
                            for t in c1:
                                if t in self.comps[c2].goal:
                                    del self.comps[c2].goal[t]

                    except Exception as e_:
                        print(e_)
                        print("")
                        # MyLogger.add_message("Update goal error!", __name__, "ERROR", "exc_info=True")
                        my_logger.error("Update goal error!", exc_info=True)
                        raise e_

                    #################################
                    # reprioritze
                    # Adds overhead
                    #################################
                    if cfg.Algorithm.reprioritize_after_sp:
                        if self.comps[c1].goal:
                            self.comps[c1].reprioritize()

                        if self.comps[c2].goal:
                            self.comps[c2].reprioritize()  

                        # Delete respective components from node_queue
                        if c1 in self.node_queue.elements:
                            self.node_queue.delete(c1)
                        if c2 in self.node_queue.elements:
                            self.node_queue.delete(c2)
                    #################################


                    ############################################
                    # ## End update destination list and rep ###
                    # ##########################################  

                    self.path_queue.put((c1,c2), UFeas)


                    my_logger.debug("Added path to path_queue")
                    my_logger.info("path_queue len now: {}".format(len(self.path_queue.elements)))

    @abstractmethod
    def tree_update(self): 
        """ Empty the path_queue if possible, then update the solution set `S`

        For each possible path, we perform a `merge` function on the components connected by this path.
        The resulting merged component is stored in `comps` and the former components are deleted.
        As soon as a solution set is updated, we allow for a `tree_check` in the future.

        """
        pass
        
    def tree_check(self):
        """When at least one solution has been added to the solution set, check to see if we have a complete tree"""

        my_logger.info("Performing tree_check")
        my_logger.info("paths in solution set: {}".format(len(self.results['dist'])))

        if cfg.Animation.visualize:
        #     # Don't plot every thing for large graphs
        #     if np.mod(self.run_debug, np.ceil(self.graph.node_count()/5000))==0:
        #         AnimateV2.update()
                AnimateV2.update()

        if self.FLAG_STATUS_PATH_CONVERGED:
           
            # Check tree size
            if len(self.results['sol']) == len(self.terminals)-1:
                # Algorithm has finished
                self.FLAG_STATUS_COMPLETE_TREE = True
                totalLen = sum(np.array(self.results['dist']))

                my_logger.info("Finished: {}".format(totalLen))

                # Add expanded node stats
                self.results['stats']['expanded_nodes'] = MultiSearch.total_expanded_nodes
                # Reset or "close" Class variables

                # Add additional stats (don't forget to reset classes)
                self.results['stats']['fcosts_time'] = sum(MyTimer.timeTable["fcosts_time"])

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
                    # ax.axis('equal')
                    # ax.set_aspect('equal', 'box')

                    plt.draw()
                    plt.pause(1)

            self.FLAG_STATUS_PATH_CONVERGED = False

    def update_global_bound(self, comp_ind: tuple):
        """The global bound depends on the exact path criteria used

        """
        # local_bound = max([2*self.comps[comp_ind].gmin, self.comps[comp_ind].fmin])
        self.global_bound_queue.put(comp_ind, self.local_bound_value(comp_ind))

    @abstractmethod
    def p_costs_func(self, search:MultiSearch, cost_to_come: dict, next: tuple)->float:
        """An abstract method used to calculate the priority cost of a node in the open set. Must be overriden! 
        
        Typically, this is the sum of g and h, i.e. f=g+h, but may depend on the user's case
        
        """
        # MUST KEEP TRACK OF FCOSTS !!!!!!!!!
        # The implementation should call 
        # super().p_costs_func(search, cost_to_come, next)

        # propagate heuristic cost from parent using BPMX for inconsistent heuristic
        if cfg.Algorithm.use_bpmx:
            h = self.h_costs_func(search, next)
            # parent node
            parent_node = search.parent[next]
            # propagated h value
            parent_h_prop = search.f[parent_node] - cost_to_come[parent_node] - search.graph.cost(parent_node, next)
            # h = max(h, search.f[parent_node] - cost_to_come[parent_node] - search.graph.cost(parent_node, next))            
            if parent_h_prop > h:
                h = parent_h_prop
        else:
            h = self.h_costs_func(search, next)

        search.f[next] = cost_to_come[next] + h

        # User must return a float

    def h_costs_func(self, search: MultiSearch, next: tuple)->float:
        """Implementation of the nearest neighbor heuristic.      
        Heuristic costs for the node 'next', neighboring 'current'

        Parameters:
            next (tuple): The node in the neighborhood of 'current' to be considered 
            component (MultiSearch): Generic Search class object (get access to all its variables)

        Info:
            h_i(u) = min{h_j(u)}  for all j in Destination(i), and for some node 'u'
        
        NOTE:
            User needs to override this if not using any heuristic

        """
        # If we don't have any goals...
        if not search.goal:
            return 0

        # Nearest neighbor heuristic
        hju = list(map(lambda goal: Heuristics.heuristic_func_wrap(next=next, goal=goal), search.goal.values()))

        # minH = min(hju)
        # minInd = hju.index(minH)
        # minGoal = search.goal[list(search.goal)[minInd]]
        hju = list(map(lambda goal: (Heuristics.heuristic_func_wrap(next=next, goal=goal), goal), search.goal.values()))
        minH, minGoal = min(hju)

        # Set current Goal
        search.currentGoal = minGoal

        # scale heuristic value
        return cfg.Algorithm.hFactor*minH 

    @abstractmethod
    def local_bound_value(self, comp_ind: tuple)->float:
        """This function must be implemented very carefully!
        
        Return the local bound value of the current comp. The user
        needs to implement this based on the shortest path criteria chosen.
        
        """
        pass

    @abstractmethod
    def shortest_path_check(self, comps_colliding:List[tuple], path_cost:float)->bool:
        """Applies a necessary condition to check whether two colliding components possess the `shortest possible path`
        between them. Speficailly, the necessary condition is based on Nicholson's Bidirectional Search paper.

        Just because the two components collide (i.e. their closed sets touch), does not mean any complete path
        between them is necessarily the shortest. 
        
        Params:
            comps_colliding: A list of component id's for which have just colliding
            path_cost: cost of a feasible path (i.e. path between any terminal)
        
        """
        pass

    def process_path_queue(self, *args, **kwargs)->list:
        """Process path queue to get paths in order of increasing cost.
        Must respect global min though!

        """
        sol = []
        while not self.path_queue.empty():
            path_cost, comps_ind = self.path_queue.get_min()

            # check global bound
            # if path_cost <= self.global_bound_queue.get_min()[0]:
            # to avoid numerical issues we have to check for tolerance!
            rhs = self.global_bound_queue.get_min()[0]
            if abs(path_cost - rhs)<1e-9 or path_cost<=rhs:
                # append path to sol to be added
                sol.append({'path_cost': path_cost, 'comps_ind': comps_ind})

                # pop 
                self.path_queue.get()
            else:
                break

        return sol


    ''' 'where the magic happens' function '''
    def run_algorithm(self):
        """Query the algorithm to return a Steiner tree.

        The algorithm will terminate when there is a minimum-spanning tree in the metric completion of T (terminals).

        Returns:
            True: if algorithm has terminated successfully else False
        
        Resource:
            Accurate time keeping: https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python
        
        """

        # fmin, rmin, lmin, gmin
        # self.bound_tests = [[],[],[],[]]
        if cfg.Misc.profile_frame:
            cpr = cProfile.Profile()
            cpr.enable()  

        #start time
        startLoop = timer()
        while(not self.FLAG_STATUS_COMPLETE_TREE):

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
        self.results['stats']['time'] = endLoop - startLoop
        self.results['stats']['iterations'] = self.run_debug

        if cfg.Misc.profile_frame:
            cpr.disable()
            cpr.print_stats(sort='cumtime')

        # DEBUG
        # for a in zip(self.bound_tests[0], self.bound_tests[1], self.bound_tests[2], self.bound_tests[3]): print(a)

        # Store each functions time taken (see above)
        listOfFuncs = ["nominate()_time", "update()_time", "path_check()_time",\
        "tree_update()_time", "tree_check()_time" ]
        for n in listOfFuncs:
            self.results['stats'][n] = sum(MyTimer.timeTable[n])

        # Reset classes (find a better way to to do this)
        MultiSearch.reset()
        MyTimer.reset()

        # Sucessfully generated a tree     
        return True

    ########################################################################
    #   DEBUGGING ONLY
    ########################################################################

    def debug_get_comp_current_node(self, comp):
        """Return min node from frontier without purging it """
        return comp.frontier.get_min()

    def debug_global_min(self):
        return Common.path_queue_criteria(self.comps, 0, True)

    def debug_UFeas(self):
        """Print out sorted path lengths from UFeas

        """
        paths = {}
        for u in self.UFeasPath:
            for v in self.UFeasPath[u]:
                paths[(u,v)]= self.UFeasPath[u][v][0]
        # sort
        paths = {k: paths[k] for k in sorted(paths, key=lambda x: paths[x] ) }
        for k, v in paths.items():
            print(k, v)

    def debug_path_queue(self):
        """Print out sorted path lengths from path queue

        """
        paths = {k: self.UFeasPath[self.findset.get(k[0], k[0])][self.findset.get(k[1], k[1])]  for k in sorted(self.path_queue.elements)}
        
        
        # for k in sorted(self.path_queue.elements):
        #     set1 = self.findset.get(k[0], k[0])
        #     set2 = self.findset.get(k[1], k[1])

        #    print(set1, set2)

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
            if minVal is None or j.fmin < minVal:
                minVal= j.fmin
        my_logger.debug("comp: {}, terminals: {}, fmin: {}".format(i, [self.terminals[k] for k in i], j.fmin))
        return minVal

    def debug_gmin(self):
        # Log g-values
        my_logger.debug("SHOWING GMIN VALUES")
        my_logger.debug("GGGGGGGGGGGGGGGGGGG")
        minVal = None
        for i,j in self.comps.items(): 
            # MyLogger.add_message("comp: {}, terminals: {}, gmin: {}".format(i, [self.terminals[k] for k in i], j.gmin), __name__, "DEBUG")
            if minVal is None or j.gmin < minVal:
                minVal= j.gmin

        my_logger.debug("comp: {}, terminals: {}, gmin: {}".format(i, [self.terminals[k] for k in i], j.gmin))
        return minVal

    def debug_pmin(self):
        # Log p-values
        my_logger.debug("SHOWING PMIN VALUES")
        my_logger.debug("PPPPPPPPPPPPPPPPPPP")
        minVal = None
        for i,j in self.comps.items(): 
            # MyLogger.add_message("comp: {}, terminals: {}, pmin: {}".format(i, [self.terminals[k] for k in i], j.pmin), __name__, "DEBUG")
            if minVal is None or j.pmin < minVal:
                minVal= j.pmin

        my_logger.debug("comp: {}, terminals: {}, pmin: {}".format(i, [self.terminals[k] for k in i], j.pmin))

        return minVal

    def debug_lmin(self):
        my_logger.debug("SHOWING LMIN VALUES")
        my_logger.debug("LLLLLLLLLLLLLLLLLLL")
        minVal = None
        for i,j in self.comps.items(): 
            # MyLogger.add_message("comp: {}, terminals: {}, lmin: {}".format(i, [self.terminals[k] for k in i], j.lmin), __name__, "DEBUG")
            if minVal is None or j.lmin < minVal:
                minVal= j.lmin
        
        my_logger.debug("comp: {}, terminals: {}, lmin: {}".format(i, [self.terminals[k] for k in i], j.lmin))

        return minVal

    def debug_rmin(self):
        my_logger.debug("SHOWING RMIN VALUES")
        my_logger.debug("RRRRRRRRRRRRRRRRRRR")
        minVal = None
        for i,j in self.comps.items(): 
            # MyLogger.add_message("comp: {}, terminals: {}, rmin: {}".format(i, [self.terminals[k] for k in i], j.rmin), __name__, "DEBUG")
            if minVal is None or j.rmin < minVal:
                minVal = j.rmin
        my_logger.debug("comp: {}, terminals: {}, rmin: {}".format(i, [self.terminals[k] for k in i], j.rmin))
        return minVal    

    