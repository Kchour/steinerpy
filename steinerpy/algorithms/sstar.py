import steinerpy.config as cfg
from steinerpy.framework import Framework
from .common import Common


class SstarHS(Framework):
    """S* Merged with Heuristics (A*) """

    def __init__(self, G, T):
        super().__init__(G, T)

    def p_costs_func(self, component, cost_so_far, next):
        """fcost(n) = gcost(n) + hcost(n, goal)
        
        Parameters:
            component (GenericSearch): Generic Search class object (get access to all its variables)
            cost_so_far (dict): Contains all nodes with finite g-cost
            next (tuple): The node in the neighborhood of 'current' to be considered 

        Returns:
            fcost (float): The priority value for the node 'next'

        """
        h = self.h_costs_func(next, component) 
        fCost= cost_so_far[next] +  h

        # Keep track of Fcosts
        component.f[next] = fCost
        
        return fCost

    def h_costs_func(self, next, object_):
        """Heuristic costs for the node 'next', neighboring 'current'

        Parameters:
            next (tuple): The node in the neighborhood of 'current' to be considered 
            component (GenericSearch): Generic Search class object (get access to all its variables)

        Info:
            h_i(u) = min{h_j(u)}  for all j in Destination(i), and for some node 'u'
        
        """
        # If we don't have any goals...
        if not object_.goal:
            return 0
  
        # need to look at current object's destination...which changes
        # hju = list(map(lambda goal: htypes(type_, next, goal), terminals))
        # hju = list(map(lambda goal: htypes(type_, next, goal), [terminals[i] for i in comps[object_.id]['destinations']]))
        # hju = list(map(lambda goal: htypes(type_, next, goal), [dest for dest in object_.goal]))
        # hju = list(map(lambda goal: Common.grid_based_heuristics(type_=type_, next=next, goal=goal), object_.goal.values()))
        hju = list(map(lambda goal: Common.heuristic_func_wrap(next=next, goal=goal), object_.goal.values()))

        minH = min(hju)
        minInd = hju.index(minH)
        minGoal = object_.goal[list(object_.goal)[minInd]]

        # Set current Goal
        object_.currentGoal = minGoal

        return minH

    # See Common class
    # def path_queue_criteria(self, comps, path_distance, comp_edge):
    #     """ Check to see if candidate path is shorter than estimates. Override if needed 
        
    #     Returns:
    #         True: if candidate path is shorter than every other path
    #         False: otherwise

    #     """

class SstarBS(Framework):
    def __init__(self, G, T):
        super().__init__(G, T)

    def p_costs_func(self, component, cost_so_far, next):
        """fcost(n) = gcost(n) + hcost(n, goal)        
        
        Parameters:
            component (GenericSearch): Generic Search class object (get access to all its variables)
            cost_so_far (dict): Contains all nodes with finite g-cost
            next (tuple): The node in the neighborhood of 'current' to be considered 

        Returns:
            fcost (float): The priority value for the node 'next'

        """
        # fCost= cost_so_far[next] +  self.h_costs_func(next, component) 
        fCost =  cost_so_far[next]

        # Keep track of Fcosts
        component.f[next] = fCost

        return fCost

    def h_costs_func(self, next, object_):
        # return 0
        pass

    def shortest_path_check(self, comps, term_edge, bestVal):
        # import itertools as it

        t1, t2 = term_edge
        comp1 = comps[t1]
        comp2 = comps[t2]
        eps = 1         # cheapest edge cost

        # a1, a2 = comp1.frontier.get_test()[0], comp2.frontier.get_test()[0]
        # b1, b2 = comp1.minimum_radius(), comp2.minimum_radius()
        # c1, c2 = comp1.currentF, comp2.currentF

        # pp = it.combinations((a1,a2,b1,b2,c1,c2), 2)

        # if bestVal <= comp1.gmin() + comp2.gmin():
        # if bestVal <= comp1.currentF + comp2.currentF:
        if bestVal <= comp1.gmin + comp2.gmin:
            # if bestVal < sum(min(pp)):
            # if bestVal <= comp1.minimum_radius() + comp2.minimum_radius() + 1:
            return True
        else:
            return False

class SstarMM(Framework):
    """Meet-in-the-Middle implementation with heuristics """
    def __init__(self, G, T):
        super().__init__(G, T)

    def p_costs_func(self, component, cost_so_far, next):
        """Normally, fcost(n) = gcost(n) + hcost(n, goal), but this function 
            can be used very generically to define the priority of node 'next'        
        
        Parameters:
            component (GenericSearch): Generic Search class object (get access to all its variables)
            cost_so_far (dict): Contains all nodes with finite g-cost
            next (tuple): The node in the neighborhood of 'current' to be considered 

        Returns:
            priority (float): The priority value for the node 'next'

        """
        # fCost= cost_so_far[next] +  self.h_costs_func(next, component) 
        g_next = cost_so_far[next]
        fCost =  g_next + self.h_costs_func(next, component) 
        # fCost =  g_next 

        # Keep track of Fcosts
        component.f[next] = fCost

        # From MM paper
        priority = max(fCost, 2*g_next)
        return priority

    def h_costs_func(self, next, object_):
        """Heuristic costs for the node 'next', neighboring 'current'

        Parameters:
            next (tuple): The node in the neighborhood of 'current' to be considered 
            component (GenericSearch): Generic Search class object (get access to all its variables)

        Info:
            h_i(u) = min{h_j(u)}  for all j in Destination(i), and for some node 'u'
        
        """
        # If we don't have any goals...
        if not object_.goal:
            return 0
  
        # need to look at current object's destination...which changes
        # hju = list(map(lambda goal: Common.grid_based_heuristics(type_=type_, next=next, goal=goal), object_.goal.values()))
        hju = list(map(lambda goal: Common.heuristic_func_wrap(next=next, goal=goal), object_.goal.values()))
        
        minH = min(hju)
        minInd = hju.index(minH)
        minGoal = object_.goal[list(object_.goal)[minInd]]

        # Set current Goal
        object_.currentGoal = minGoal

        return minH

    def shortest_path_check(self, comps, term_edge, bestVal):
        # import itertools as it

        t1, t2 = term_edge
        comp1 = comps[t1]
        comp2 = comps[t2]
        # eps = 1        # cheapest edge cost HARD-CODED 

        # a1, a2 = comp1.frontier.get_test()[0], comp2.frontier.get_test()[0]
        # b1, b2 = comp1.minimum_radius(), comp2.minimum_radius()
        # c1, c2 = comp1.currentF, comp2.currentF

        # pp = it.combinations((a1,a2,b1,b2,c1,c2), 2)
        C = min(comp1.pmin, comp2.pmin)
    
        # from paper
        # if bestVal <= max(C, comp1.currentF, comp2.currentF, \
        #     comp1.g[comp1.current]+comp2.g[comp2.current] + eps):
        if bestVal <= max(C, comp1.fmin, comp2.fmin, \
            comp1.gmin + comp2.gmin):
            # if bestVal < sum(min(pp)):
            # if bestVal <= comp1.minimum_radius() + comp2.minimum_radius() + 1:
            return True
        else:
            return False  

class SstarMM0(Framework):
    """Meet-in-the-Middle implementation without heuristics (Brute-force search)"""
    def __init__(self, G, T):
        super().__init__(G, T)

    def p_costs_func(self, component, cost_so_far, next):
        """Normally, fcost(n) = gcost(n) + hcost(n, goal), but this function 
            can be used very generically to define the priority of node 'next'        
        
        Parameters:
            component (GenericSearch): Generic Search class object (get access to all its variables)
            cost_so_far (dict): Contains all nodes with finite g-cost
            next (tuple): The node in the neighborhood of 'current' to be considered 

        Returns:
            priority (float): The priority value for the node 'next'

        TODO:
            * Rename this function!

        """
        # fCost= cost_so_far[next] +  self.h_costs_func(next, component) 
        g_next = cost_so_far[next]
        fCost =  g_next 

        # Keep track of Fcosts
        component.f[next] = fCost

        # From MM paper
        priority = max(fCost, 2*g_next)
        return priority

    def h_costs_func(self, next, object_):
        """Heuristic costs for the node 'next', neighboring 'current'

        Parameters:
            next (tuple): The node in the neighborhood of 'current' to be considered 
            component (GenericSearch): Generic Search class object (get access to all its variables)

        Info:
            h_i(u) = min{h_j(u)}  for all j in Destination(i), and for some node 'u'
        
        """
        pass

    def shortest_path_check(self, comps, term_edge, bestVal):
        # import itertools as it

        t1, t2 = term_edge
        comp1 = comps[t1]
        comp2 = comps[t2]
        eps = 1         # cheapest edge cost HARD-CODED 

        # a1, a2 = comp1.frontier.get_test()[0], comp2.frontier.get_test()[0]
        # b1, b2 = comp1.minimum_radius(), comp2.minimum_radius()
        # c1, c2 = comp1.currentF, comp2.currentF

        # pp = it.combinations((a1,a2,b1,b2,c1,c2), 2)
        C = min(comp1.pmin, comp2.pmin)
    
        # from paper
        if bestVal <= max(C, comp1.fmin, comp2.fmin, \
            comp1.gmin + comp2.gmin):
            # if bestVal < sum(min(pp)):
            # if bestVal <= comp1.minimum_radius() + comp2.minimum_radius() + 1:
            return True
        else:
            return False                      