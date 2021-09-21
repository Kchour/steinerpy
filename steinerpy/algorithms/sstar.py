"""Implementation of S* variants here, based on merged vs. unmerged,
    and different path criteria's

"""
from typing import List


from steinerpy.library.search.search_algorithms import MultiSearch
# import steinerpy.config as cfg
from steinerpy.algorithms.merged import Merged
from steinerpy.algorithms.unmerged import Unmerged
from steinerpy.common import PathCriteria
  
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
        # keep track of 
        g_next = cost_so_far[next]
        fCost =  g_next + self.h_costs_func(search, next) 

        # Keep track of Fcosts
        search.f[next] = fCost

        # From MM paper
        priority = max(fCost, 2*g_next)
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
        g_next = cost_so_far[next]
        fCost =  g_next 

        # Keep track of Fcosts
        search.f[next] = fCost

        # From MM paper
        priority = max(fCost, 2*g_next)
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
