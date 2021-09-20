"""Common utility functions exposed as static methods 

Todo:
    * Add support for generic graphs 
"""

import numpy as np
import logging

import steinerpy.config as cfg
from steinerpy.library.search.search_utils import reconstruct_path

my_logger = logging.getLogger(__name__)


class Common:

    @staticmethod    
    def solution_handler(comps, path_queue, cycle_detector, terminals, criteria, merging=False, use_depots=False):
        """ Handle shortest paths between terminals, but delay adding solutions if certain critera are not met

        Parameter:
            comps (dict): `GenericSearch` objects keyed by indices
            path_queue (PriorityQueueHeap): a minimum priority queue with paths as items
            cycle_detector (CycleDetector): An object used to detect cycles
            terminals (list): a list of tuples of terminals
            criteria (bool): True if path criteria is satisifed (in fact the shortest wrt all paths) 
            merging (bool): True if we are using the merge function (i.e. using class `Framework` or Sstar)
            
        Note:
            pathQueue.elements: ({'terms':self.t1+self.t2, 'path':path, 'selData':self.selData, 'selNode': self.selNode, 'dist':dist}, dist)

        """

        sol = []
        while not path_queue.empty():
            # get the min fcost path
            poppedQ = path_queue.get_min()
            dist,comps_ind = poppedQ
                    
            if not merging:
                # Check for cycle only, don't explicity add the edge
                iscycle = cycle_detector.add_edge(*comps_ind, test=True)
            else: 
                # if merge
                iscycle = False               

            if not iscycle:
                # Check tree criteria to know when to add path 
                if criteria(comps=comps, path_distance = dist): 
                    if not merging:
                        # if not a cycle, then add the edge!
                        cycle_detector.add_edge(*comps_ind)
                        
                        # 
                        if cfg.Algorithm.reprioritize_after_merge:
                            findset = cycle_detector.parent_table[comps_ind[0]]
                            new_goals = {i: terminals[i] for i in set(range(len(terminals)))-set(findset)}
                            for c in findset:
                                comps[(c,)].goal = new_goals
                                comps[(c,)].reprioritize()

                    
                    # Add solution
                    # path = poppedQ[1]['path']
                    # dist = poppedQ[1]['dist']
                    # edge = poppedQ[1]['term_actual']
                    # terms = poppedQ[1]['terms']
                    sol.append({'dist':dist, 'components':comps_ind})
                    my_logger.debug("Added poppedQ path to sol!") 

                    # remove next least cost edge
                    path_queue.get()

                else:
                    # path_queue.put(poppedQ[1], poppedQ[0])
                    break
            else:
                # remove the edge that induces a cycle
                path_queue.get()
        return sol 

    @staticmethod
    def get_path(comps, sel_node, term_edge, reconstruct_path_func):
        """Generate path between two components (specifically, its closest terminals)

        Parameters:
            comps: GenericSearch objects
            sel_node (tuple): The nominated node
            sel_data (dict): Information pertaining to the nominated node. See `Framework` class for more information
        
        Return:
            path (numpy.ndarray), dist (float), term_actual (tuple): Returned items for `Framework`
        
        """
        t1,t2 = term_edge
        # Build path given a linked list
        pathA = reconstruct_path_func(comps[t1].parent, start=None, goal=sel_node, order="forward" )
        pathB = reconstruct_path_func(comps[t2].parent, start=None, goal=sel_node, order="reverse")

        path = []
        path.extend(pathA)
        path.extend(pathB[1:])

        # get dist by using gcosts
        distA = comps[t1].g[sel_node]
        distB = comps[t2].g[sel_node]
        dist = distA + distB

        term_actual = (tuple({pathA[0]}),tuple({pathB[-1]}))
        return path, dist, term_actual
    
    @staticmethod
    def add_solution(path, dist, edge, solution_set, terminals):
        """Add paths to our Steiner Tree.

        Parameters:
            path (numpy.ndarray): A complete path for a 2d grid graph
            dist (float): Length of a path
            edge (tuple): Actual terminals (end points) in a path
            solution_set (dict): The Steiner tree
            terminals (list): A list of all the terminal nodes

        """
        if len(solution_set['sol']) < len(terminals)-1:
            solution_set['dist'].append(dist)
            solution_set['path'].append(path)
            solution_set['sol'].append(edge)
            my_logger.debug("Added edge no.: {}".format(len(solution_set['sol']))) 


    @staticmethod
    def merge_comps(comps, term_edge, nodeQueue, cache):  
        """ Merge function handler

        Parameters:
            comps (dict): A dictionary of `GenericSearch` objects, keyed by indices.
            term_edge (tuple): The components involved (FIXME not terminals, but components!)
            nodeQueue (PriorityQueue): A min priority queue of nominated components, we have to modify this after merge
            cache (dict): The working forest or cache of closed nodes. We also have to modify this after merge

        """
        t1,t2 = term_edge
        # merge two different comps, delete non-merged comps respectively
        mergedComp = comps.get(t1) + comps.get(t2)        
        # mergedComp = comp1 + comp2
        comps[mergedComp.id] = mergedComp
        del comps[t1]
        del comps[t2]
        
        # Delete old references from every location
        if t1 in nodeQueue:
            nodeQueue.delete(t1)
        if t2 in nodeQueue:
            nodeQueue.delete(t2) #Is this the right way to do this?

    @staticmethod
    def create_search_objects(search_class, graph, p_costs_func, frontier_type, terminals, visualize=False):
        """ Register `GenericSearch` class objects with an id, so we can do multiple searches and merge them

        Parameters:
            search_class (GenericSearch): See `GenericSearch` class for more information
            graph (SquareGrid, MyGraph): Our graph, which the search algorithm is performed on
            p_costs_func: A scalar function, returning the priority costs of a node (based on heuristics)
            frontier_type: Allows user to select the type (a class) of priority queue to use
            terminals (list): A list of terminal tuples
            visualize (bool): whether to visualize the algorithm while in-progress, or not

        Returns:
            {(index, ):  search_class(graph, f_costs_func, start, frontierType=frontier_type(), \
                    goal={i: terminals[i] for i in set(range(len(terminals)))-set((index,))}, visualize=visualize, id=index) \
                    for index,start in enumerate(terminals) }
        """

        return {(index, ):  search_class(graph, start, {i: terminals[i] for i in set(range(len(terminals)))-set((index,))},\
                     p_costs_func, frontierType=frontier_type(), visualize=visualize, id=index) \
                    for index,start in enumerate(terminals) }

    @staticmethod
    def subsetFinder(comp, comp_list):
        """Method for finding the subsets an elemen belongs to

        Parameters:
            comps (iter): The items we want to find subsets for (iterable)
            comp_list (iter): The list of subsets (iterable)

        """
        subs = []
        for c in comp:
            for cl in comp_list:
                if set(c).issubset(set(cl)):
                    subs.append(cl)
        return subs




