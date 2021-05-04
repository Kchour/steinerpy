import numpy as np

from steinerpy.framework import Framework
from .common import Common
import steinerpy.config as cfg
from steinerpy.library.animation import AnimateV2
from steinerpy.library.search.generic_algorithms import GenericSearch
from steinerpy.library.search.search_utils import reconstruct_path, CycleDetection
from steinerpy.library.logger import MyLogger

class Astar(Framework):
    def __init__(self, graph, terminals):
        Framework.__init__(self, graph, terminals)
        # self.terminals = T
        # self.graph = G
        # self.S = {'sol':[], 'dist':[], 'path':[]}
        # self.FLAG_STATUS_completeTree = False
        
        # cycle detection required in Astar unmerged
        self.cd = CycleDetection([(t,) for t in range(len(terminals))])

    def nominate(self):
        """Each component nominates a node from its open set
               
        For each component in `comps`, we nominate a single node. 
        If the nominate function returns a `True`, we move on to
        storing the nominated node and its priority value (Fcost)
        in a queue variable called nodeQueue.
        
        """
        MyLogger.add_message("performing nominate() ", __name__, "INFO")

        for ndx, c in self.comps.items():
            if ndx not in self.nodeQueue.elements:
                if c.nominate():
                    self.nodeQueue.put(ndx, c.currentP)

    # Optional: Check for collisions and update destination list before update
    def update(self):
        """Update the component pertaining to the least fcost-valued node
        
        Among all nominated nodes, choose the one with least f cost using a priority queue

        """
        # Get best ndx from priority queue
        # print(self.run_debug)
        MyLogger.add_message("performing update() ", __name__, "INFO")

        # if not self.nodeQueue.empty():
        best_priority, best_ndx = self.nodeQueue.get()  
        
        # get best component object, and g cost of best node
        bestC = self.comps[best_ndx]
        bestCurrent = bestC.current
        bestGVal = bestC.g[bestCurrent]  
        bestFVal = bestC.currentF
        
        # Get parent (if possible)
        bestParent = bestC.parent.get(bestCurrent, None)

        # Store and return the selected node(s)
        self.selNode = bestCurrent
        self.selData = {} 
        self.selData.update({'to': bestParent, 'terminalInd': best_ndx, 'gcost': bestGVal, 'fcost':bestFVal, 'status':'closed'})


        MyLogger.add_message("updated {} with node {}".format(best_ndx, bestCurrent), __name__, "Debug")


        ##############################
        ### NEW FUNCTIONALITY HERE ###
        ##############################
        # destination list
        # dest = self.comps[self.selData['terminalInd']].goal.values()

        # # Check for complete path
        # self.iscollided = Common.set_collision_check(sel_node=self.selNode, sel_data=self.selData,\
        #     target_list=dest, cache=self.F)

        # if self.iscollided:
        #     # Define terminal indices
        #     t1 = self.selData['terminalInd']
        #     t2 = (self.terminals.index(self.selNode), ) 

        #     MyLogger.add_message("goals(PRE) of {} is {}".format(t1, self.comps[t1].goal), __name__, "Debug")
        #     MyLogger.add_message("goals(PRE) of {} is {}".format(t2, self.comps[t2].goal), __name__, "Debug")
            
        #     # update destination list TEST THIS
        #     del self.comps[t1].goal[t2[0]]
        #     del self.comps[t2].goal[t1[0]]

        #     # reprioritize
        #     if self.comps[t2].goal:
        #         self.comps[t2].reprioritize()

        #     MyLogger.add_message("goals(POST) of {} is {}".format(t1, self.comps[t1].goal), __name__, "Debug")
        #     MyLogger.add_message("goals(POST) of {} is {}".format(t2, self.comps[t2].goal), __name__, "Debug")


        ###############################
        ###  END NEW FUNCTIONALITY  ###
        ###############################

        # Now update the closed/open list for the 'best' component
        # make sure goals are not empty!
        # if self.comps[best_ndx].goal:
        bestC.update()

        # if cfg.visualize:
        #     self.plotCurrent.update_clean(self.selNode)
        # print('selected Node: ', self.selNode)
        # print('selected Data: ', self.selData)

    # override path_check because target_list is different
    def path_check(self):     
        """  
        1. Check for set collision
        2. Find shortest path among two colliding sets

        """    
        # MyLogger.add_message("performing path_check() ", __name__, "INFO")
        # t1 = self.selData['terminalInd']
        # updatedComp = self.comps[t1]
        # for t2, c in self.comps.items():
        #     # skip if trying to intersect with itself
        #     if t1 == t2:
        #         continue
            
        #     updateSet = set(updatedComp.current)
        #     # # Add updated component recent open and closed list
        #     # updateSet = set(updatedComp.currentNeighs)
        #     # # updateSet = set(updatedComp.frontier.elements)
        #     # updateSet.add(updatedComp.current)
        #     # if c.currentNeighs:
        #     #     updateSet.union(set(c.currentNeighs))
        #     # updateSet.add(c.current)


        #     UFeas = None
        #     for k in updateSet:
        #         if k in self.comps[t1].goal.values():
        #             candU = self.comps[t1].g[k]
        #             if  UFeas is None or candU < UFeas:
        #                 UFeas = candU
        #                 commonNode = k
           
        #     # if UFeas is not None:
        #     #     updateSet = set(updatedComp.frontier.elements)
        #     #     for k in updateSet:
        #     #         if k in self.comps[t1].g and k in self.comps[t2].g:
        #     #             candU = self.comps[t1].g[k] + self.comps[t2].g[k]
        #     #             if  UFeas is None or candU < UFeas:
        #     #                 UFeas = candU
        #     #                 commonNode = k

        #     ######################################################################################    
        #     # try:
        #     #     jointSet = ((c.g[k] + updatedComp.g[k], k) for k in set(c.g).intersection(set(updatedComp.frontier.elements)))
        #     #     minItem = min(jointSet)
        #     #     # UFeas = minItem[0]
        #     #     # commonNode = minItem[1]
        #     #     del jointSet
        #     # except:
        #     #     # UFeas, commonNode = None, None
        #     #     pass
        #     ######################################################################################
        #     # if abs(UFeas - 11.656) < 1e-3:
        #     #     self.testone.append(UFeas)

        UFeas = None
        t1 = self.selData['terminalInd']
        k = self.comps[t1].current
        if k in self.comps[t1].goal.values():
            t2 = (self.terminals.index(k),)
            if (t1,t2) not in self.pathQueue.entry_table or (t2,t1) not in self.pathQueue.entry_table: 
                UFeas = self.comps[t1].g[k]
                commonNode = k

                # set lmins for each comp
                # self.comps[t1].lmin = UFeas
                # self.comps[t2].lmin = UFeas

                #update feasible path table
                if t1 in self.UFeasPath:
                    self.UFeasPath[t1].update({t2: [UFeas, commonNode]})
                else:
                    self.UFeasPath.update({t1: {t2: [UFeas, commonNode]}})
                
                if t2 in self.UFeasPath:
                    self.UFeasPath[t2].update({t1: [UFeas, commonNode]})
                else:                           
                    self.UFeasPath.update({t2: {t1: [UFeas, commonNode]}})  

        if UFeas is not None:
            # set lmins for each component
            if UFeas < self.comps[t1].lmin or self.comps[t1].lmin == 0:
                self.comps[t1].lmin = UFeas
            if UFeas < self.comps[t2].lmin or self.comps[t2].lmin == 0:
                self.comps[t2].lmin = UFeas
            # Subtract some slack due to numerical issues
            # t1, t2 = t1feas, t2feas
            sp = self.shortest_path_check(comps=self.comps, term_edge=(t1,t2), bestVal=UFeas)
            if sp:
                # if criteria is satisfied, update the path queue               
                # Get path based on sel_node

                # path, dist, term_actual = Common.get_path(comps=self.comps, sel_node=commonNode, term_edge=(t1,t2),\
                #     reconstruct_path_func = reconstruct_path)

                ###########################################
                ### # update destination list TEST THIS ###
                ###########################################
                # MyLogger.add_message("goals(PRE) of {} is {}".format(t1, self.comps[t1].goal), __name__, "Debug")
                # MyLogger.add_message("goals(PRE) of {} is {}".format(t2, self.comps[t2].goal), __name__, "Debug")
   
                # update destination list TEST THIS
                del self.comps[t1].goal[t2[0]]
                del self.comps[t2].goal[t1[0]]

                # reprioritize
                if self.comps[t2].goal:
                    self.comps[t2].reprioritize()
                if self.comps[t1].goal:
                    self.comps[t1].reprioritize()
                # # for ndx, c in self.comps.items(): print(ndx, c.fmin_heap.pq, "\n")

                # Delete respective components from nodeQueue
                if t1 in self.nodeQueue.elements:
                    self.nodeQueue.delete(t1)
                if t2 in self.nodeQueue.elements:
                    self.nodeQueue.delete(t2)     

                # MyLogger.add_message("goals(POST) of {} is {}".format(t1, self.comps[t1].goal), __name__, "Debug")
                # MyLogger.add_message("goals(POST) of {} is {}".format(t2, self.comps[t2].goal), __name__, "Debug")

                ############################################
                # ## End update destination list and rep ###
                # ##########################################  

                MyLogger.add_message("paths in solution set: {}".format(len(self.S['dist'])), __name__, "INFO")

                # # Set another lower bound on components due to declared shortest path
                # if dist > 0 and dist < self.comps[t1].lmin or self.comps[t1].lmin == 0:
                #     self.comps[t1].lmin = dist

                # if dist > 0 and dist < self.comps[t2].lmin or self.comps[t2].lmin == 0:
                #     self.comps[t2].lmin = dist

                # self.comps[t1].lmin = dist
                # self.comps[t2].lmin = dist            
    
                # self.pathQueue.put({'terms': (t1,t2), 'term_actual': term_actual, 'path':path, 'dist':dist}, dist)
                # self.pathQueue.put( ((t1,t2), term_actual, tuple(path), dist), dist)
                self.pathQueue.put((t1,t2), UFeas)

                MyLogger.add_message("Added path to pathQueue", __name__, "DEBUG")

                MyLogger.add_message("pathQueue len now: {}".format(len(self.pathQueue.elements)), __name__, "INFO")

                if cfg.Misc.console_level == "DEBUG":
                    self.debug_fmin()
                    self.debug_gmin()
                    self.debug_pmin()
                    # self.debug_lmin()
                    self.debug_rmin()
                    testtesttest=1
        # # destination list
        # # dest = self.comps[self.selData['terminalInd']].goal.values()

        # # # Check for complete path
        # # self.iscollided = Common.set_collision_check(sel_node=self.selNode, sel_data=self.selData,\
        # #     target_list=dest, cache=self.F)  

        # if self.iscollided:
        #     # Define terminal indices
        #     t1 = self.selData['terminalInd']            
        #     t2 = (self.terminals.index(self.selNode), )      #### THIS CHANGED TOO #### 

        #     # # update destination list
        #     # del self.comps[t1].goal[t2[0]]
        #     # del self.comps[t2].goal[t1[0]]

        #     # # reprioritize
        #     # if self.comps[t2].goal:
        #     #     self.comps[t2].reprioritize()
            
        #     # Get path based on sel_node
        #     path, dist, term_actual = Common.get_path(comps=self.comps, sel_node=self.selNode, term_edge=(t1,t2),\
        #          reconstruct_path_func = reconstruct_path)

        #     # Logging information
        #     MyLogger.add_message("Collision between {},{}".format(t1, t2), __name__, "Debug")
        #     MyLogger.add_message("Feasible Path dist {}".format(dist), __name__, "Debug")
        #     MyLogger.add_message("sel node n {}".format(self.selNode), __name__, "Debug")
        #     MyLogger.add_message("{} gcosts(n) {}".format(t1, self.comps[t1].g[self.selNode] ), __name__, "Debug")
        #     MyLogger.add_message("{} gcosts(n) {}".format(t2, self.comps[t2].g[self.selNode]), __name__, "Debug")
        #     MyLogger.add_message("{} current f {}".format(t1, self.comps[t1].currentF), __name__, "Debug")
        #     MyLogger.add_message("{} current f {}".format(t2, self.comps[t2].currentF), __name__, "Debug")

        #     # Update the path Queue, but don't add duplicates
        #     # print(self.run_debug)
        #     # if (t1,t2) not in [p[2]['terms'] for p in self.pathQueue.elements]: 
        #     self.pathQueue.put({'terms': (t1,t2), 'term_actual': term_actual, 'path':path, 'selData':self.selData, 'selNode': self.selNode, 'dist':dist}, dist)

    def tree_update(self):
        """override tree_update because we need cycle detection and no merging """
        MyLogger.add_message("performing tree_update() ", __name__, "INFO")

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
                raise ValueError("distances don't match! path queue and feasible table is conflicting!", self.terminals, self, pdist, dist)

            Common.add_solution(path=path, dist=dist, edge=term_actual,\
                solution_set=self.S, terminals=self.terminals)

            # True as soon as we add a solution
            self.FLAG_STATUS_pathConverged = True

            # Perform merging
            # Common.merge_comps(self.comps, term_edge=(t1,t2), nodeQueue=self.nodeQueue, cache=self.F)

            # TODO find a better way to animate path
            if cfg.Animation.visualize:
                # self.animateS.update_clean(np.vstack(self.S['path']).T.tolist())
                AnimateV2.add_line("solution", np.vstack(self.S['path']).T.tolist(), 'ro', markersize=10, zorder=10, alpha=0.5)

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
        # type_ = 'diagonal_nonuniform'
        type_ = cfg.Algorithm.sstar_heuristic_type
  
        # need to look at current object's destination...which changes
        # hju = list(map(lambda goal: htypes(type_, next, goal), terminals))
        # hju = list(map(lambda goal: htypes(type_, next, goal), [terminals[i] for i in comps[object_.id]['destinations']]))
        # hju = list(map(lambda goal: htypes(type_, next, goal), [dest for dest in object_.goal]))
        hju = list(map(lambda goal: Common.grid_based_heuristics(type_=type_, next=next, goal=goal), object_.goal.values()))
        
        if hju:
            minH = min(hju)
        else:
            minH = 0
        #minInd = hju.index(minH)

        return minH

    def debug_all_fmin(self):
        for c in self.comps.values():
            print(c.id, c.fmin)
    