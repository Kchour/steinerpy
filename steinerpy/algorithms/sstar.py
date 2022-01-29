"""Implementation of S* variants here, based on merged vs. unmerged,
    and different path criteria's

"""
import logging
from typing import List
from steinerpy.heuristics import Heuristics


from steinerpy.library.search.search_algorithms import MultiSearch
# import steinerpy.config as cfg
from steinerpy.algorithms.merged import Merged
from steinerpy.algorithms.unmerged import Unmerged
from steinerpy.common import PathCriteria

logger = logging.getLogger(__name__) 

##########################################################
#   MERGED 
##########################################################

class SstarHS(Merged):
    """S* Merged with Heuristics (A*) """

    def p_costs_func(self, search:MultiSearch, cost_so_far: dict, next: tuple):
        """Priority function for nodes in the frontier
        
        Parameters:
            search (MultiSearch): Generic Search class object (get access to all its variables)
            cost_so_far (dict): Contains all nodes with finite g-cost
            next (tuple): The node in the neighborhood of 'current' to be considered 

        Returns:
            fcost (float): The priority value for the node 'next'

        """
        super().p_costs_func(search, cost_so_far, next)
        
        # return fcost
        return search.f[next]

    def shortest_path_check(self, comps_colliding:List[tuple], path_cost:float)->bool:
        """ Check to see if candidate path is a confirmed shortest path.
        
        Returns:
            True: if candidate path is a confirmed shortest path between c1, c2
            False: otherwise

        """
        # Get component objects
        _c1, _c2 = comps_colliding
        c1 = self.comps[_c1]
        c2 = self.comps[_c2]

        return PathCriteria.path_criteria_pohl(path_cost, c1, c2)

    def local_bound_value(self, comp_ind: tuple)->float:
        """This function must be implemented very carefully
        
        """
        return  self.comps[comp_ind].fmin


class SstarBS(Merged):

    def p_costs_func(self, search:MultiSearch, cost_so_far: dict, next: tuple):
        """fcost(n) = gcost(n) + hcost(n, goal)        
        
        Parameters:
            component (MultiSearch): Generic Search class object (get access to all its variables)
            cost_so_far (dict): Contains all nodes with finite g-cost
            next (tuple): The node in the neighborhood of 'current' to be considered 

        Returns:
            fcost (float): The priority value for the node 'next'

        """        
        super().p_costs_func(search, cost_so_far, next)
        # return g cost
        return cost_so_far[next]

    def h_costs_func(self, search: MultiSearch, next: tuple):
        return 0

    def shortest_path_check(self, comps_colliding:List[tuple], path_cost:float)->bool:
        """ Check to see if candidate path is a confirmed shortest path.
        
        Returns:
            True: if candidate path is a confirmed shortest path between c1, c2
            False: otherwise

        """
        # Get component objects
        _c1, _c2 = comps_colliding
        c1 = self.comps[_c1]
        c2 = self.comps[_c2]

        return PathCriteria.path_criteria_nicholson(path_cost, c1, c2)

    def local_bound_value(self, comp_ind: tuple)->float:
        return 2*self.comps[comp_ind].gmin

class SstarMM(Merged):
    """Meet-in-the-Middle implementation with heuristics """

    def p_costs_func(self, search:MultiSearch, cost_so_far: dict, next: tuple):
        """Normally, fcost(n) = gcost(n) + hcost(n, goal), but this function 
            can be used very generically to define the priority of node 'next'        
        
        Parameters:
            component (MultiSearch): Generic Search class object (get access to all its variables)
            cost_so_far (dict): Contains all nodes with finite g-cost
            next (tuple): The node in the neighborhood of 'current' to be considered 

        Returns:
            priority (float): The priority value for the node 'next'

        """
        # # keep track of 
        # fCost =  g_next + self.h_costs_func(search, next) 

        # # Keep track of Fcosts
        # search.f[next] = fCost
        super().p_costs_func(search, cost_so_far, next)

        # From MM paper
        priority = max(search.f[next], 2*cost_so_far[next])
        return priority

    def shortest_path_check(self, comps_colliding:List[tuple], path_cost:float)->bool:
        # Get component objects
        _c1, _c2 = comps_colliding
        c1 = self.comps[_c1]
        c2 = self.comps[_c2]

        eps = 1         # cheapest edge cost TODO: need to detect cheapest edge cost

        return PathCriteria.path_criteria_mm(path_cost, c1, c2)

    def local_bound_value(self, comp_ind: tuple)->float:
        return max([2*self.comps[comp_ind].gmin, self.comps[comp_ind].fmin, self.comps[comp_ind].pmin])

class SstarMM0(Merged):
    """Meet-in-the-Middle implementation without heuristics (Brute-force search)"""
    def __init__(self, G, T):
        super().__init__(G, T)

    def p_costs_func(self, search:MultiSearch, cost_so_far: dict, next: tuple):
        """Normally, fcost(n) = gcost(n) + hcost(n, goal), but this function 
            can be used very generically to define the priority of node 'next'        
        
        Parameters:
            component (MultiSearch): Generic Search class object (get access to all its variables)
            cost_so_far (dict): Contains all nodes with finite g-cost
            next (tuple): The node in the neighborhood of 'current' to be considered 

        Returns:
            priority (float): The priority value for the node 'next'

        TODO:
            * Rename this function!

        """
        super().p_costs_func(search, cost_so_far, next)

        # From MM paper
        priority = max(search.f[next], 2*cost_so_far[next])
        return priority

    def h_costs_func(self, search: MultiSearch, next: tuple):
        """Heuristic costs for the node 'next', neighboring 'current'

        Parameters:
            next (tuple): The node in the neighborhood of 'current' to be considered 
            component (MultiSearch): Generic Search class object (get access to all its variables)

        Info:
            h_i(u) = min{h_j(u)}  for all j in Destination(i), and for some node 'u'
        
        """
        pass

    def shortest_path_check(self, comps_colliding:List[tuple], path_cost:float)->bool:
        # Get component objects
        _c1, _c2 = comps_colliding
        c1 = self.comps[_c1]
        c2 = self.comps[_c2]

        eps = 1         # cheapest edge cost TODO: need to detect cheapest edge cost

        return PathCriteria.path_criteria_mm(path_cost, c1, c2)
              

    def local_bound_value(self, comp_ind: tuple)->float:
        return max([2*self.comps[comp_ind].gmin, self.comps[comp_ind].fmin, self.comps[comp_ind].pmin])

#################################################################
# UNMERGED
#################################################################

class SstarHSUN(Unmerged):
    """S* path condition based on Pohl
    
    """

    def p_costs_func(self, search: MultiSearch, cost_to_come: dict, next: tuple) -> float:
        super().p_costs_func(search, cost_to_come, next)

        return search.f[next]

    def shortest_path_check(self, comps_colliding: List[tuple], path_cost: float) -> bool:
        # Get component objects
        _c1, _c2 = comps_colliding
        c1 = self.comps[_c1]
        c2 = self.comps[_c2]

        return PathCriteria.path_criteria_pohl(path_cost, c1, c2)

    def local_bound_value(self, comp_ind: tuple)->float:
        """This function must be implemented very carefully
        
        """
        return  self.comps[comp_ind].fmin

class SstarBSUN(Unmerged):

    def p_costs_func(self, search: MultiSearch, cost_to_come: dict, next: tuple) -> float:
         super().p_costs_func(search, cost_to_come, next)   
         return cost_to_come[next]

    def h_costs_func(self, search: MultiSearch, next: tuple)->float:
        return 0

    def shortest_path_check(self, comps_colliding: List[tuple], path_cost: float) -> bool:
        # Get component objects
        _c1, _c2 = comps_colliding
        c1 = self.comps[_c1]
        c2 = self.comps[_c2]

        return PathCriteria.path_criteria_nicholson(path_cost, c1, c2)  

    def local_bound_value(self, comp_ind: tuple)->float:
        return 2*self.comps[comp_ind].gmin

class SstarMMUN(Unmerged):
    
    def p_costs_func(self, search: MultiSearch, cost_to_come: dict, next: tuple) -> float:
        super().p_costs_func(search, cost_to_come, next)
        return max(search.f[next], 2*cost_to_come[next])

    def shortest_path_check(self, comps_colliding: List[tuple], path_cost: float) -> bool:
        # Get component objects
        _c1, _c2 = comps_colliding
        c1 = self.comps[_c1]
        c2 = self.comps[_c2]

        return PathCriteria.path_criteria_mm(path_cost, c1, c2)

    def local_bound_value(self, comp_ind: tuple)->float:
        return max([2*self.comps[comp_ind].gmin, self.comps[comp_ind].fmin, self.comps[comp_ind].pmin])
        # return max([2*self.comps[comp_ind].gmin, self.comps[comp_ind].fmin]) 

class SstarMM0UN(Unmerged):
    
    def p_costs_func(self, search: MultiSearch, cost_to_come: dict, next: tuple) -> float:
        super().p_costs_func(search, cost_to_come, next)
        return max(search.f[next], 2*cost_to_come[next])

    def h_costs_func(self, search: MultiSearch, next: tuple) -> float:
        return 0

    def shortest_path_check(self, comps_colliding: List[tuple], path_cost: float) -> bool:
        # Get component objects
        _c1, _c2 = comps_colliding
        c1 = self.comps[_c1]
        c2 = self.comps[_c2]

        return PathCriteria.path_criteria_mm(path_cost, c1, c2)

    def local_bound_value(self, comp_ind: tuple)->float:
        return max([2*self.comps[comp_ind].gmin, self.comps[comp_ind].fmin, self.comps[comp_ind].pmin])

#######################################################
# Repeat of the above but with lb-propagation #########
#######################################################

def lb_prop_func(search: MultiSearch, next: tuple) ->float:
        """Compute the h cost of node 'next' (or u) 
        using lb-propagation (Shperberg 2019)

        """
        if not search.goal:
            return 0

        # forward heuristic is the typical nearest-neighbor one
        hju = list(map(lambda goal: Heuristics.heuristic_func_wrap(next=next, goal=goal), search.goal.values()))
        minH = min(hju)
        minInd = hju.index(minH)
        minGoal = search.goal[list(search.goal)[minInd]]

        f_forward = search.g[next] + minH 

        # now try lb-propagation (involves both forward and backward heuristics)
        # loop over all components except this one
        lb = float('inf')
        for idx, comp in search.siblings.items():
            # skip self
            if idx == search.id:
                continue
            
            # # only do lb-propagation between nearest-neighbor
            # if minGoal not in comp.start:
            #     continue

            # loop over all nodes in the open set
            for item in list(comp.frontier):
                _,_,v = item
                # best lower bound between two different search fronts
                # ----------------------------------------------------
                # try recompute backward heuristic?
                hju = list(map(lambda goal: Heuristics.heuristic_func_wrap(next=v, goal=goal), comp.goal.values()))
                if hju:
                    minH = min(hju)
                else:
                    minH = 0
                f_backward = comp.g[v] + minH 

                # with respect to current forward direction's root only!
                # f_backward = comp.g[v] + Heuristics.heuristic_func_wrap(next=v, goal=search.root[next])

                # # dynamic update data structures?
                # comp.f[v] = f_backward
                # # comp.fmin_heap.put(v, f_backward)
                # # comp.frontier.put(v, f_backward)
                temp = max(f_forward, f_backward, search.g[next] + comp.g[v])

                # dont recompute backward h?
                # temp = max(f_forward, comp.f[v], search.g[next] + comp.g[v])

                # keep the minimum lower bound
                if temp < lb:
                    lb = temp

        # this is from the paper
        h = lb - search.g[next]
        logger.debug("Heuristic value, search_id: {} {}".format(h, search.id))
        return h

class SstarMMLP(SstarMM):

    def h_costs_func(self, search: MultiSearch, next: tuple) -> float:
        return lb_prop_func(search, next)

class SstarHSLP(SstarHS):

    def h_costs_func(self, search: MultiSearch, next: tuple) -> float:
        return lb_prop_func(search, next)

#### TRY THE SAME THING WITH UNMERGED VARIANTS ###

class SstarMMUNLP(SstarMMUN):

    def h_costs_func(self, search: MultiSearch, next: tuple) -> float:
        return lb_prop_func(search, next)

class SstarHSUNLP(SstarHSUN):

    def h_costs_func(self, search: MultiSearch, next: tuple) -> float:
        return lb_prop_func(search, next)