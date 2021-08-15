"""Common utility functions exposed as static methods 

Todo:
    * Add support for generic graphs 
"""

import numpy as np
import logging

# from steinerpy.library.logger import MyLogger # deprecated
import steinerpy.config as cfg

my_logger = logging.getLogger(__name__)

class Common:

    @staticmethod
    def set_collision_check(sel_node, sel_data, target_list, cache):
        """Check for path completeness between two terminals

        Parameters:
            sel_node (tuple): The nominated node 
            sel_data (dict): The nominated data
            target_list: Used to determine when two sets have collided       
            cache (dict): The working forest of all the closed nodes so far

        Returns:
            True: if sets have collided (using the target_list)
            False: otherwise
        
        """
        if sel_node not in target_list:
            # No complete path yet, update cache
            cache[sel_node] = sel_data
            return False
        elif sel_node in target_list and sel_data['terminalInd'] == cache[sel_node]['terminalInd']: 
            # DEBUG CASE to allow running until no more nodes available. Don't allow self collisions
            # FIXED update cache with lower valued gcost 
            # if sel_data['gcost'] < cache[sel_node]['gcost']:
            #     cache[sel_node] = sel_data
            return False
        elif sel_node in target_list:
            # A complete path exists, 
            # FIXED update cache with lower valued gcost
            # if sel_data['gcost'] < cache[sel_node]['gcost']:
            #     cache[sel_node] = sel_data
            # return False
            return True

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
        
        # if cfg.console_level == "DEBUG":
        #     for ndx, k in enumerate(path_queue.elements):
        #         MyLogger.add_message("path {}: {}".format(ndx, k[2]['path']), __name__, "Debug")

        sol = []
        # FLAG_STATUS_pathConverged = False
        while not path_queue.empty():
            # get the min fcost path
            poppedQ = path_queue.get()
            # terms_ind, terms_actual, path, dist = poppedQ[1]
            dist,comps_ind = poppedQ
                    
            # # MyLogger.add_message("testing poppedQ dist: {}, comps: {}, actualTs: {}"\
            # #     .format(poppedQ[1]['dist'], poppedQ[1]['terms'], poppedQ[1]['term_actual']), __name__, "DEBUG")
            # MyLogger.add_message("testing poppedQ dist: {}, comps: {}, actualTs: {}"\
            #     .format(dist, terms_ind, terms_actual), __name__, "DEBUG")        

            # MyLogger.add_message("testing poppedQ path: {}".format(path), __name__, "DEBUG")

            if not merging:
                # Check for cycle only, don't explicity add the edge
                iscycle = cycle_detector.add_edge(*comps_ind, test=True)
            else: 
                # if merge
                iscycle = False               

            if iscycle:
                break
            else:
                # if not a cycle, then add the edge!
                if not merging:
                    cycle_detector.add_edge(*comps_ind)
                # Check tree criteria to know when to add path 
                minPathF = criteria(comps=comps, path_distance = dist)
                
                if minPathF: 
                    if not merging:
                        # # Update the destination list, priority of components with path added between them
                        # for ndx, c in comps.items():
                            
                        #     disjointSet = [list(i) for i in cycle_detector.indices if set(ndx).issubset(set(i))][0]
                        #     c.goal = {i: terminals[i] for i in set(range(len(terminals)))-set(disjointSet)}
                            
                        #     # make sure goal isn't empty
                        #     if c.goal:
                        #         c.reprioritize()

                        # for set_ in cycle_detector.indices:
                        #     new_goals = {i: terminals[i] for i in set(range(len(terminals)))-set(set_)}
                        #     for c in set_:
                        #         comps[(c,)].goal = new_goals
                        #         comps[(c,)].reprioritize()

                        findset = cycle_detector.parent_table[comps_ind[0]]
                        new_goals = {i: terminals[i] for i in set(range(len(terminals)))-set(findset)}
                        for c in findset:
                            comps[(c,)].goal = new_goals
                            comps[(c,)].reprioritize()
                        # comps[comps_ind[0]].goal = new_goals
                        # comps[comps_ind[0]].reprioritize()
                        pass
                        
                    
                    # Add solution
                    # path = poppedQ[1]['path']
                    # dist = poppedQ[1]['dist']
                    # edge = poppedQ[1]['term_actual']
                    # terms = poppedQ[1]['terms']
                    sol.append({'dist':dist, 'components':comps_ind})
                    my_logger.debug("Added poppedQ path to sol!") 

                else:
                    path_queue.put(poppedQ[1], poppedQ[0])
                    break
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
        # path = np.vstack((pathA,pathB))
        path = []
        path.extend(pathA)
        path.extend(pathB)

        # get dist by using gcosts
        distA = comps[t1].g[sel_node]
        distB = comps[t2].g[sel_node]
        dist = distA + distB

        # print('terminals: ',pathA[0], pathB[-1])
        term_actual = (tuple(pathA[0]),tuple(pathB[-1]))
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
    def get_shortest_path(comps, term_edge, f_costs_func):
        """shortest path check between two components after collision.

        Parameters:
            comps (dict): A dictionary of `GenericSearch` objects, keyed by indices.
            term_edge (tuple): The components involved (FIXME not terminals, but components!)
            f_costs_func: A scalar function for computing the fcosts (FIXME probably not useful)

        Returns:
            bestVal, bestk, bestkDict: what is actually returned!
            bestVal (float): The length of a path
            bestk (tuple): A node found on the path
            bestkDict (dict): information pertaining to the best path
            
        Todo:
            * Consider passing in self, then we wont have to change anything?
            * Rename some variables

        """
        # figure out comps
        t1, t2 = term_edge
        comp1 = comps[t1]
        comp2 = comps[t2]    

        # get g costs
        g1 = comp1.g
        g2 = comp2.g

        # get intersection
        test = set(g1).intersection(set(g2))
        # get next minimal frontier costs
        # minf1 = comp1.frontier.get_test()[0]
        # minf2 = comp2.frontier.get_test()[0]

        # find path with shortest total g cost
        bestk, bestVal = None, None
        for k in test:
            if bestVal is None or g1[k] + g2[k] < bestVal:
                bestVal = g1[k] + g2[k]
                bestk = k

        # Find best Dict. Based on min G cost or max F?
        if comp2.g[bestk] < comp1.g[bestk]:
            bestkDict = {'to': comp2.parent[bestk], 'terminalInd': comp2.id, \
                'gcost': comp2.g[bestk], 'fcost': f_costs_func(comp2, comp2.g, bestk)}
        else:
            bestkDict = {'to': comp1.parent[bestk], 'terminalInd': comp1.id, \
                'gcost': comp1.g[bestk], 'fcost': f_costs_func(comp1, comp1.g, bestk)}

        return bestVal, bestk, bestkDict

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
        
        # # Take care of nodeQueue id's after merge. Consider not doing this (see issue #10)
        # Don't do this 
        # if self.t1 in self.nodeQueue.elements:
        #     self.nodeQueue.elements[mergedComp.id] = self.nodeQueue.elements[self.t1]
        # if self.t2 in self.nodeQueue.elements:
        #     self.nodeQueue.elements[mergedComp.id] = self.nodeQueue.elements[self.t2]

        # delete old id's,
        if t1 in nodeQueue.elements:
            nodeQueue.delete(t1)
        if t2 in nodeQueue.elements:
            nodeQueue.delete(t2) #Is this the right way to do this?

        # Take care of current node id
        # sel_data[sel_node]['terminalInd'] = mergedComp.id

        # debugging primal-dual: nodeQueue. Need to set the self.current!
    
        # (3,):{'node': (-15, -13), 'priority': 5.0}
        # (0,):{'node': (0, -10), 'priority': 5.0}
        # (4,):{'node': (3, 0), 'priority': 5.0}
        # (2, 1):{'node': (-15, 1), 'priority': 5.0}
        # len():4

        # # FIXME GET RID OF THIS
        # # Ensure cache id's refer to the updated components
        # for k in cache.values():
        #     if k['terminalInd'] == t1:
        #         k['terminalInd'] = mergedComp.id
        #     elif k['terminalInd'] == t2:
        #         k['terminalInd'] = mergedComp.id

    @staticmethod
    def create_search_objects(search_class, graph, f_costs_func, frontier_type, terminals, visualize=False):
        """ Register `GenericSearch` class objects with an id, so we can do multiple searches and merge them

        Parameters:
            search_class (GenericSearch): See `GenericSearch` class for more information
            graph (SquareGrid, MyGraph): Our graph, which the search algorithm is performed on
            f_costs_func: A scalar function, returning the fcosts of a node (based on heuristics)
            frontier_type: Allows user to select the type (a class) of priority queue to use
            terminals (list): A list of terminal tuples
            visualize (bool): whether to visualize the algorithm while in-progress, or not

        Returns:
            {(index, ):  search_class(graph, f_costs_func, start, frontierType=frontier_type(), \
                    goal={i: terminals[i] for i in set(range(len(terminals)))-set((index,))}, visualize=visualize, id=index) \
                    for index,start in enumerate(terminals) }
        """

        return {(index, ):  search_class(graph, f_costs_func, start, frontierType=frontier_type(), \
                    goal={i: terminals[i] for i in set(range(len(terminals)))-set((index,))}, visualize=visualize, id=index) \
                    for index,start in enumerate(terminals) }

    @staticmethod
    def path_queue_criteria(comps, path_distance):
        """ Check to see if candidate path is shorter than estimates. Override if needed

        This method helps to preserve Kruskal's property, namely that path P is only considered
        if and only if all paths cheaper than P have already been considerd! 
        
        Returns:
            True: if candidate path is shorter than every other path
            False: otherwise

        """
        # for c in comps.values():
        #     if path_distance > c.fmin and len(comps)>2:
        #         return False

        # return True
        rmin1 = None
        rmin2 = None
        lmin = None
        fmin = None
        # include a small parameter to overcome floating point error
        eps = 1e-6
        # gmin1 = None
        # gmin2 = None
        # do minmax global lower bound
        min_g_lb = np.inf
        for c in comps.values():
            # rmin2 = rmin1
            # rmin1_c = c.rmin
            # # gmin1_c = c.gmin
            # # gmin2 = gmin1

            # if rmin1 is None or rmin1_c < rmin1:
            #     rmin1 = rmin1_c
            # # find best non-zero lmin
            # if (lmin is None or c.lmin < lmin) and c.lmin > 0:
            #     lmin = c.lmin

            # if fmin is None or c.fmin < fmin:
            #     fmin = c.fmin
            
            # # if gmin1 is None or gmin1_c < gmin1:
            # #     gmin1 = c.gmin
            if c.lmin == 0:
                lh = 2*c.gmin
            else:
                lh = c.lmin

            # DEBUG PRINT
            c_lb = max(c.fmin, 2*c.rmin, lh)
            if c_lb < min_g_lb:
                min_g_lb = c_lb
            # print(path_distance, c.id, c.fmin, c.rmin, c.lmin, c_lb)
            # print(c.id, c.fmin, path_distance)
        # print(min_g_lb)
        # # handle case where every lmin is 0
        # if lmin is None:
        #     lmin = 0

        # # cannot add any path when lmin is 0 
        # if lmin == 0:
        #     return False

        # elif path_distance > max(fmin, 2*rmin1, lmin)+eps:
        # if path_distance > fmin:

        # print(path_distance, min_g_lb)

        if path_distance > min_g_lb+eps:
                # if path_distance > c.fmin:
                # if path_distance > 2*min(minF1, minF2):    
                return False
        return True

    @staticmethod
    def shortest_path_check(comps, term_edge, bestVal):
        """After set collision, check to see if we can find the shortest possible path between the sets

        Parameter:
            comps (dict): A dictionary of `GenericSearch` objects, keyed by indices.
            term_edge (tuple): The components involved (FIXME not terminals, but components!)
            bestVal (float): The current best valued path between two sets. May not represent the actual shortest path.

        Returns:
            True: if we satisfy the path convergence criteria
            False: otherwise
        """
        t1,t2 = term_edge
        comp1 = comps[t1]
        comp2 = comps[t2]    

        # This is Nicholson's Criteria
        # if bestVal <= comp1.g[comp1.current] + comp2.g[comp2.current]:
        # if bestVal < comp1.currentF + comp2.currentF:    
        # if bestVal <= max(comp1.currentF, comp2.currentF): 
        if bestVal <= max(comp1.fmin, comp2.fmin):
            # Shortest path is 
            return True
        else:
            #skip merging if this is the case
            return False

    @staticmethod
    def custom_heuristics(next, goal):
        """Allow user to define heuristic
        
        Parameters:
            next (tuple):
            goal (tuple):

        """
        pass

    @staticmethod
    def grid_based_heuristics(type_, next, goal):
            """ Heuristics for a flat grid graph

            Parameters:
                type_ (str): the type of heuristics to use (manhattan, euclidean, diagonal_uniform, diagonal_nonuniform)
                next (tuple): The source vertex  
                goal (tuple): The destination vertex

            Todo:
                * Add support for 3D graphs

            """
            try: 
                (x1, y1) = next
                (x2, y2) = goal
                if type_ == 'manhattan':
                    heuristic = abs(x1 - x2) + abs(y1 - y2)
                elif type_ == 'euclidean':
                    v = [x2 - x1, y2 - y1]
                    heuristic = np.hypot(v[0], v[1])
                elif type_ == 'diagonal_uniform':
                    heuristic = max(abs(x1 - x2), abs(y1 - y2))
                elif type_ == 'diagonal_nonuniform':
                    dmax = max(abs(x1 - x2), abs(y1 - y2))
                    dmin = min(abs(x1 - x2), abs(y1 - y2))
                    heuristic = 1.414*dmin + (dmax - dmin)
                elif type_ == 'preprocess':
                    from steinerpy.library.pipeline import GenerateHeuristics
                    heuristic = GenerateHeuristics.heuristic_wrap(next, goal)     
                elif type_ == 'custom':
                    heuristic = Common.custom_heuristics(next, goal)
                else:
                    raise Exception("Please select an available heuristic type: 'manhattan', 'euclidean', 'diagonal_uniform', 'diagonal_nonuniform', 'preprocess'")
            except Exception as e:
                raise e

            return cfg.Algorithm.hFactor*heuristic 

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


 