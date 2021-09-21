"""Common utility functions exposed as static methods 

Todo:
    * Add support for generic graphs 
"""

from steinerpy.library.search.search_algorithms import MultiSearch
import numpy as np
import logging

import steinerpy.config as cfg
from steinerpy.library.search.search_utils import reconstruct_path

my_logger = logging.getLogger(__name__)


class Common:

    @staticmethod
    def get_path(c1:MultiSearch, c2: MultiSearch, common_node:tuple):
        """Generate path between two components (specifically, its closest terminals)

        Parameters:
            comps: MultiSearch objects
            sel_node (tuple): The nominated node
            sel_data (dict): Information pertaining to the nominated node. See `Framework` class for more information
        
        Return:
            path (numpy.ndarray), dist (float), term_actual (tuple): Returned items for `Framework`
        
        """
        # Build path given a linked list
        pathA = c1.reconstruct_path(goal=common_node, start=None, order="forward" )
        pathB = c2.reconstruct_path(goal=common_node, start=None, order="reverse")

        path = []
        path.extend(pathA)
        path.extend(pathB[1:])

        # get dist by using gcosts
        distA = c1.g[common_node]
        distB = c2.g[common_node]
        dist = distA + distB


        # TODO make this neater get rid of commas if able
        term_actual = (tuple({pathA[0]}),tuple({pathB[-1]}))
        return path, dist, term_actual
    
    @staticmethod
    def add_solution(path, dist, edge, results, terminals):
        """Add paths to our Steiner Tree.

        Parameters:
            path (numpy.ndarray): A complete path for a 2d grid graph
            dist (float): Length of a path
            edge (tuple): Actual terminals (end points) in a path
            results (dict): The Steiner tree
            terminals (list): A list of all the terminal nodes

        """
        if len(results['sol']) < len(terminals)-1:
            results['dist'].append(dist)
            results['path'].append(path)
            results['sol'].append(edge)
            my_logger.debug("Added edge no.: {}".format(len(results['sol']))) 


    @staticmethod
    def merge_comps(comps, term_edge, nodeQueue, cache):  
        """ Merge function handler

        Parameters:
            comps (dict): A dictionary of `MultiSearch` objects, keyed by indices.
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
    def create_search_objects(search_class, graph, p_costs_func, terminals, visualize=False):
        """ Register `MultiSearch` class objects with an id, so we can do multiple searches and merge them

        Parameters:
            search_class (MultiSearch): See `MultiSearch` class for more information
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
                     p_costs_func, visualize=visualize, id=index) \
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


class PathCriteria:

    @staticmethod
    def path_criteria_interface(path_cost: float, c1: MultiSearch, c2: MultiSearch)->bool:
        """Example here: user's path criteria must follow this pattern.
        
        """

    @staticmethod
    def path_criteria_pohl(path_cost, c1: MultiSearch, c2: MultiSearch)->bool:
        if path_cost <= max(c1.fmin, c2.fmin):
            return True
        else:
            return False

    @staticmethod
    def path_criteria_nicholson(path_cost, c1: MultiSearch, c2: MultiSearch)->bool:
        # This is Nicholson's criteria
        if path_cost <= c1.gmin + c2.gmin:
            # shortest path confirmed
            return True
        else:
            # shortest path not confirmed
            return False

    @staticmethod
    def path_criteria_mm(path_cost, c1: MultiSearch, c2: MultiSearch)->bool:
        # from MM paper
        C = min(c1.pmin, c2.pmin)
        if path_cost <= max(C, c1.fmin, c2.fmin, c1.gmin + c2.gmin):
            return True
        else:
            return False  