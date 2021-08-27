import numpy as np
import logging

from steinerpy.framework import Framework
from .common import Common
import steinerpy.config as cfg
from steinerpy.library.animation import AnimateV2
from steinerpy.library.search.search_utils import reconstruct_path, CycleDetection
# from steinerpy.library.logger import MyLogger

my_logger = logging.getLogger(__name__)

class Unmerged(Framework):
    def __init__(self, graph, terminals):
        Framework.__init__(self, graph, terminals)
        # self.terminals = T
        # self.graph = G
        # self.S = {'sol':[], 'dist':[], 'path':[]}
        # self.FLAG_STATUS_completeTree = False
        
        # cycle detection required in Astar unmerged
        self.cd = CycleDetection([(t,) for t in range(len(terminals))])

    def tree_update(self):
        """override tree_update because we need cycle detection and no merging """
        my_logger.info("Performing Tree Update")
        # Empty path queue, gather up the solutions
        sol = Common.solution_handler(comps=self.comps, path_queue=self.pathQueue, cycle_detector=self.cd, \
            terminals=self.terminals, criteria=self.path_queue_criteria, merging=False)

        # add paths to solution set
        for s in sol:
            # self.add_solution(s['path'], s['dist'], s['terms'])
            # t1,t2 = s['terms']
             # if t1 in self.comps and t2 in self.comps:
            # print(s)

            t1,t2 = s['components']
            pdist = s['dist']

            # Get common node between two components
            dist, commonNode = self.UFeasPath[t1][t2]

            path, _, term_actual = Common.get_path(comps=self.comps, sel_node=commonNode, term_edge=(t1,t2),\
                reconstruct_path_func = reconstruct_path)
            
            if abs(pdist-dist)>0.1 or abs(pdist-_)>0.1:
                # print("")
                raise ValueError("distances don't match! path queue and feasible table is conflicting!", self.terminals, self, pdist, dist, _)

            Common.add_solution(path=path, dist=dist, edge=term_actual,\
                solution_set=self.S, terminals=self.terminals)

            # True as soon as we add a solution
            self.FLAG_STATUS_pathConverged = True

            # Perform merging
            # Common.merge_comps(self.comps, term_edge=(t1,t2), nodeQueue=self.nodeQueue, cache=self.F)

            # TODO find a better way to animate path
            if cfg.Animation.visualize:
                # self.animateS.update_clean(np.vstack(self.S['path']).T.tolist())
                AnimateV2.add_line("solution", np.vstack(self.S['path']).T.tolist(), 'yo', markersize=10, zorder=10)

    # cost and heuristic function definition
    def f_costs_func(self, object_, cost_so_far, next):
        """fcost(n) = gcost(n) + hcost(n, goal)        
        
        Parameters:
            object_ (GenericSearch): Generic Search class object (get access to all its variables)
            cost_so_far (dict): Contains all nodes with finite g-cost
            next (tuple): The node in the neighborhood of 'current' to be considered 

        Returns:
            fcost (float): The priority value for the node 'next'

        """
        fCost= cost_so_far[next] +  self.h_costs_func(next, object_) 
        # fCost= cost_so_far[next] 

        # Keep track of Fcosts
        object_.f[next] = fCost

        return fCost

    def h_costs_func(self, next, object_): 
        """ 
        h_i(u) = min{h_j(u)}  for all j in Destination(i), and for some node 'u'
        
        """  
        # need to look at current object's destination...which changes
        # hju = list(map(lambda goal: htypes(type_, next, goal), terminals))
        # hju = list(map(lambda goal: htypes(type_, next, goal), [terminals[i] for i in comps[object_.id]['destinations']]))
        # hju = list(map(lambda goal: htypes(type_, next, goal), [dest for dest in object_.goal]))
        # hju = list(map(lambda goal: Common.grid_based_heuristics(type_=type_, next=next, goal=goal), object_.goal.values()))
        # hju = list(map(lambda goal: Common.grid_based_heuristics(type_=type_, next=next, goal=goal), object_.goal.values()))
        hju = list(map(lambda goal: Common.heuristic_func_wrap(next=next, goal=goal), object_.goal.values()))

        
        if hju:
            minH = min(hju)
        else:
            minH = 0
        #minInd = hju.index(minH)

        return minH

    def debug_all_fmin(self):
        for c in self.comps.values():
            print(c.id, c.fmin)
    