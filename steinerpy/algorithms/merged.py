"""Implementation of the Merged class, subclassing Framework, where 2 components are merged
    when minimum shortest path is confirmed between them. Several book-keeping operations are done
    such as creating an entirely new component id and removing all references to the old separate ids.
    This involves modifying the node_queue, global_bound_queue, and UFeas table.

    Key difference between merged and unmerged algorithms is in the tree_update() function
    
"""

import logging
import numpy as np

import steinerpy.config as cfg
from steinerpy.framework import Framework
from steinerpy.library.animation import AnimateV2
from steinerpy.common import Common
# from steinerpy.library.search.search_utils import reconstruct_path
# from steinerpy.library.search.search_algorithms import MultiSearch

# configure and create logger
my_logger = logging.getLogger(__name__) 

class Merged(Framework):
    
    def tree_update(self):
        my_logger.info("performing tree_update")

        # my_logger.debug("global kruskal value {}".format( Common.path_queue_criteria(self.comps, 0, True) ) )

        # Empty path queue, gather up the solutions in solution queue (FIFO)
        # solQueue = Common.solution_handler(comps=self.comps, path_queue=self.path_queue, cycle_detector=None, \
        #     terminals=self.terminals, criteria=self.path_queue_criteria, merging=True, use_depots=self.use_depots)
        
        solQueue = self.process_path_queue()

        my_logger.info("solQueue len: {}".format(len(solQueue)))

        # add paths to solution set
        for ndx, s in enumerate(solQueue):
            # self.add_solution(s['path'], s['dist'], s['terms'])
            # t1,t2 = s['terms']
            # my_logger.debug("emptying solQueue iter: {}".format(ndx+1))

            my_logger.debug("adding edge with value {}".format(s['path_cost']))

            # t1, t2 = Common.subsetFinder(s['terms'], self.comps)
            # MyLogger.add_message("Inspecting path {}. Old Comps {}. New Comps {}. Terminals {}. length {}".format(s['path'], s['terms'], (t1,t2), s['term_actual'], s['dist']), __name__, "DEBUG")

            # THESE ARE INDICIES
            c1,c2 = s['comps_ind']     #could be old components
            # To avoid adding redundant paths. 

            pdist = s['path_cost']
            
            # find set t1 and t2 belong to
            if c1[0] in self.findset: 
                c1 = self.findset[c1[0]]
            if c2[0] in self.findset:
                c2 = self.findset[c2[0]]
            
            # helps detect cycles...so we don't have to clear out pathQueue
            if c1 == c2:
                continue

            # debug bounds
            # self.debug_bounds(t1)
            # self.debug_bounds(t2)
        
            # update findset (TODO: make sure keys are tuples!)
            for c in (c1+c2):
                self.findset[c] = (c1+c2)
              
            # Get common node between two components
            dist, common_node = self.UFeasPath[c1][c2]

            if abs(pdist-dist)>0.1:
                # print("")
                my_logger.warning("inconsistent edge between terminals (may be due to inadmissible h?): {} {}".format(t1, t2))

                # This may be due to inadmissible heuristic?
                # raise ValueError("distances don't match! path queue and feasible table is conflicting!", self.terminals, self, pdist, dist)

            # reconstruct path
            path, _, term_actual = Common.get_path(self.comps[c1], self.comps[c2], common_node)

            self.sol_edges.add((c1,c2))
            self.sol_edges.add((c2,c1))

            try:              
                # Add solution
                Common.add_solution(path=path, dist=dist, edge=term_actual,\
                results=self.results, terminals=self.terminals)

                my_logger.debug("Just added path no. {}. Terminals {}".format(len(self.results['dist']), term_actual))

                # True as soon as we add a solution
                self.FLAG_STATUS_PATH_CONVERGED = True

                # Perform merging
                # merge two different comps, delete non-merged comps respectively
                mergedComp = self.comps[c1] + self.comps[c2]        
                # self.comps[mergedComp.id] = mergedComp
                # del self.comps[c1]
                # del self.comps[c2]

                
                # Delete old references from every location (nodeQueue, pathQueue, globalQueue)
                if c1 in self.node_queue:
                    self.node_queue.delete(c1)
                if c2 in self.node_queue:
                    self.node_queue.delete(c2) 

                if c1 in self.global_bound_queue:
                    self.global_bound_queue.delete(c1)
                if c2 in self.global_bound_queue:
                    self.global_bound_queue.delete(c2)
            
                # update path keys which reference old comp ids
                # TODO: This is not that straightforward
                if c1 in self.path_queue:
                    self.path_queue.delete(c1)
                if c2 in self.path_queue:
                    self.path_queue.delete(c2)

                # Update feasible path keys and subkeys to refer to updated component indices                
                # minimize over exclusive union
                set1 = set(self.UFeasPath[c1])              # components adjacent to compt1
                set2 = set(self.UFeasPath[c2])              # components adjacent to compt2
                ex_union = set1.union(set2)-set({c1,c2})    # all adjacent components excluding comp1,comp2
                merged_key = {c1+c2: {}}
                delList = []
                for k in ex_union:
                    # Create adjacency list for the newly merged component:
                    # For components adjacent to both comp1 and comp2 indvidually, retain the shorter feasible path
                    # elif for components adjacent to only comp1 or comp2, just take the feas path directly
                    if k in set1 and k in set2:
                        # retain the shorter path to the merged component
                        if self.UFeasPath[c1][k][0] < self.UFeasPath[c2][k][0]:
                            merged_key[c1+c2].update({k: [self.UFeasPath[c1][k][0], self.UFeasPath[c1][k][1]]})      
                        else:
                            merged_key[c1+c2].update({k: [self.UFeasPath[c2][k][0], self.UFeasPath[c2][k][1]]})       
                    elif k in set1:
                        # simply take adjacency of comp1
                        merged_key[c1+c2].update({k: self.UFeasPath[c1][k]})
                    elif k in set2:
                        # simply take the adjacency of comp2
                        merged_key[c1+c2].update({k: self.UFeasPath[c2][k]})


                    # update old sub-keys to point to merged comp
                    # if kth-component was adjacent to t1 or t2, update its adj list
                    if c1 in self.UFeasPath[k] and c2 in self.UFeasPath[k]:
                        # make sure to take minimum! Dont blindly set this
                        if self.UFeasPath[k][c1] < self.UFeasPath[k][c2]:
                            self.UFeasPath[k].update({c1+c2: self.UFeasPath[k][c1]})
                        else:
                            self.UFeasPath[k].update({c1+c2: self.UFeasPath[k][c2]})
                        delList.append((k,c1))
                        delList.append((k,c2))
                    elif c1 in self.UFeasPath[k]:
                        self.UFeasPath[k].update({c1+c2: self.UFeasPath[k][c1]})
                        delList.append((k,c1))
                        # del self.UFeasPath[k][t1]
                    elif c2 in self.UFeasPath[k]:
                        self.UFeasPath[k].update({c1+c2: self.UFeasPath[k][c2]})
                        delList.append((k,c2))
                        # del self.UFeasPath[k][t2]

                # delete old unmerged comps
                del self.UFeasPath[c1]
                del self.UFeasPath[c2]

                # delete old sub-keys
                for d in delList:
                    del self.UFeasPath[d[0]][d[1]]
                
                # Add merged comp
                self.UFeasPath.update(merged_key)

                # Log f costs after merging
                # MyLogger.add_message("{} current fmin {}".format(t1+t2, self.comps[t1+t2].fmin), __name__, "Debug")
    
                # See if any feasible paths in merged components can be added to path queue
                # This is important because they may get skipped!
                for n in self.UFeasPath[c1+c2]:
                    paths = self.UFeasPath[c1+c2][n]
                    # check sp
                    if self.shortest_path_check([(c1+c2),(n)], paths[0]):
                        self.path_queue.put(((c1+c2),(n)), paths[0])

                # TODO find a better way to animate path
                if cfg.Animation.visualize:
                #     self.animateS.update_clean(np.vstack(self.results['path']).T.tolist())

                #     self.plotTerminals.update(np.array(self.terminals).T.tolist())
                #     if self.graph.obstacles:
                #         self.plotObstacle.update(np.array(self.graph.obstacles).T.tolist())

                    AnimateV2.add_line("solution", np.vstack(self.results['path']).T.tolist(), 'yo', markersize=10, zorder=10)
                    # # if self.graph.obstacles:
                    pass

            except Exception as e:
                my_logger.error("Merging error!", exc_info=True)
                print(self.terminals)
                raise e

            my_logger.info("Total tree edges now: {}".format(len(self.results['dist'])))
                   
        my_logger.info("pathQueue len now: {}".format(len(self.path_queue)))



