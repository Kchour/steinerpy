"""Implementation of S* variants here, based on merged vs. unmerged,
    and different path criteria's

"""
import logging
from typing import List
from steinerpy.heuristics import Heuristics
import steinerpy.config as cfg


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
        hju = list(map(lambda goal: (Heuristics.heuristic_func_wrap(next=next, goal=goal), goal), search.goal.values()))
        minH, minGoal = min(hju)
        # minInd = hju.index(minH)
        # minGoal = search.goal[list(search.goal)[minInd]]
        # minGoal = search.goal[minInd]
        search.currentGoal = minGoal

        # forward values
        f_forward = search.g[next] + minH
        g_forward = search.g[next] 

        # loop over each component and find the minimum possible backward bounds 
        lb = float('inf')
        for id, comp in search.siblings.items():
            # skip current search id
            if id == search.id:
                continue
            # estimate optimal path cost between these two components
            est_c = max(f_forward, comp.fmin, g_forward + comp.gmin)
            # we add all nodes with f<est_c
            ready_list = []
            while not comp.fmin_heap.empty():
                # peek only
                value, node = comp.fmin_heap.get_min()
                if value <= est_c:
                    ready_list.append((comp.g[node], node))
                    # pop
                    comp.fmin_heap.get()
                else:
                    break
            ready_list.sort()
            # add removed nodes back into fmin_heap
            for item in ready_list:
                _, node = item
                comp.fmin_heap.put(node,  comp.f[node])

            # now compute backward values
            # min_back_node = ready_list[0][1]
            min_back_node = ready_list[0][1]
            g_backward = ready_list[0][0]
            f_backward = comp.f[min_back_node]
            current_lb = max(f_forward, f_backward, g_forward + g_backward)
            lb = min(current_lb, lb)

        # # find component with least f
        # min_comp = None
        # min_f_sofar = None
        # for id, comp in search.siblings.items():
        #     if id == search.id:
        #         continue
        #     if min_f_sofar is None or comp.fmin < min_f_sofar:
        #         min_comp = comp
        #         min_f_sofar = comp.fmin


        # find component minGoal belongs to 
        # for comp in search.siblings.values():
        #     if minGoal in comp.start:
        #         break

        # est_c = max(f_forward, comp.fmin, g_forward + comp.gmin)
        # # we add all nodes with f<est_c
        # ready_list = []
        # while not comp.fmin_heap.empty():
        #     # peek only
        #     value, node = comp.fmin_heap.get_min()
        #     if value <= est_c:
        #         ready_list.append((comp.g[node], node))
        #         # pop
        #         comp.fmin_heap.get()
        #     else:
        #         break
        # ready_list.sort()
        # # add removed nodes back into fmin_heap
        # for item in ready_list:
        #     _, node = item
        #     comp.fmin_heap.put(node,  comp.f[node])

        # # now compute backward values
        # # min_back_node = ready_list[0][1]
        # min_back_node = ready_list[0][1]
        # g_backward = ready_list[0][0]
        # f_backward = comp.f[min_back_node]
        # lb = max(f_forward, f_backward, g_forward + g_backward)

        # # find component with least f
        # min_comp = None
        # min_f_sofar = None
        # for id, comp in search.siblings.items():
        #     if id == search.id:
        #         continue
        #     if min_f_sofar is None or comp.fmin < min_f_sofar:
        #         min_comp = comp
        #         min_f_sofar = comp.fmin
        # comp = min_comp
        # # compute backward values using sorting trick
        # # est C
        # # est_c = max(search.fmin, comp.fmin)
        # # est_c = max(f_forward, comp.fmin)
        # est_c = 0

        # value, node = comp.fmin_heap.get()
        # ready_list = [(comp.g[node], node)]
        # while True:
        #     # consider all nodes with f <= est_c
        #     # repeatedly call this until gf + gb <= est_c
        #     # update est_c with min(ff, fb, gf + gb)
        #     # while not comp.fmin_heap.empty():
        #     if not comp.fmin_heap.empty():
        #         # peek only
        #         value, node = comp.fmin_heap.get_min()
        #         if value <= est_c:
        #             ready_list.append((comp.g[node], node))
        #             # pop
        #             comp.fmin_heap.get()
        #         # else:
        #         #     break

        #     # sort ready list by g
        #     ready_list.sort()

        #     # now compute lower bound
        #     min_back_node = ready_list[0][1]
        #     g_backward = ready_list[0][0]
        #     f_backward = comp.f[min_back_node]
        #     # assert search.g[next] + g_backward <= est_c 
        #     if search.g[next] + g_backward <= est_c:
        #         break
        #     else:
        #         # update est_c 
        #         est_c = min(f_forward, f_backward, search.g[next]+g_backward)
            
        #     if comp.fmin_heap.empty():
        #         break

        # # add popped nodes back into opposing comp's fmin_heap
        # for item in ready_list:
        #     _, node = item
        #     comp.fmin_heap.put(node,  comp.f[node])
        # lb = max(f_forward, f_backward, search.g[next]+g_backward)


        h = lb - search.g[next]
        logger.debug("Heuristic value, search_id: {} {}".format(h, search.id))
        return cfg.Algorithm.hFactor*h

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