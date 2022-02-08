"""Im

"""

import numpy as np
import logging

from steinerpy.framework import Framework
from steinerpy.common import Common
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
        
        # Separate cycle detection routine required in S*-unmerged
        self.cycle_detection = CycleDetection([(t,) for t in range(len(terminals))])

    def tree_update(self):
        """override tree_update because we need cycle detection and no merging """
        # my_logger.info("Performing Tree Update")
        # # Empty path queue, gather up the solutions
        # sol = Common.solution_handler(comps=self.comps, path_queue=self.pathQueue, cycle_detector=self.cd, \
        #     terminals=self.terminals, criteria=self.path_queue_criteria, merging=False)

        sol = self.process_path_queue()

        # add paths to solution set
        for s in sol:
            # self.add_solution(s['path'], s['dist'], s['terms'])
            # t1,t2 = s['terms']
             # if t1 in self.comps and t2 in self.comps:
            # print(s)

            c1,c2 = s['comps_ind']
            pdist = s['path_cost']

            # Get common node between two components
            dist, common_node = self.UFeasPath[c1][c2]

            path, _, term_actual = Common.get_path(self.comps[c1], self.comps[c2], common_node)

            self.sol_edges.add((c1,c2))
            self.sol_edges.add((c2,c1))

            if abs(pdist-dist)>0.1 or abs(pdist-_)>0.1:
                # print("")
                my_logger.warning("inconsistent edge between terminals (may be due to inadmissible h?): {} {}".format(c1, c2))

                # may be due to inadmissible heuristic?
                # raise ValueError("distances don't match! path queue and feasible table is conflicting!", self.terminals, self, pdist, dist, _)

            my_logger.debug("ITERATION {}".format(self.run_debug))
            my_logger.debug("solution updated with edge {} {}, cost {}".format(c1, c2, dist))

            Common.add_solution(path=path, dist=dist, edge=term_actual,\
                results=self.results, terminals=self.terminals)

            # True as soon as we add a solution
            self.FLAG_STATUS_PATH_CONVERGED = True

            # Perform merging
            # Common.merge_comps(self.comps, term_edge=(t1,t2), nodeQueue=self.nodeQueue, cache=self.F)

            # TODO find a better way to animate path
            if cfg.Animation.visualize:
                # self.animateS.update_clean(np.vstack(self.results['path']).T.tolist())
                AnimateV2.add_line("solution", np.vstack(self.results['path']).T.tolist(), 'yo', markersize=10, zorder=10)

    def process_path_queue(self) -> list:
        """Process path queue 

        """
        sol = []
        while not self.path_queue.empty():
            path_cost, comps_ind = self.path_queue.get_min()

            # check for cycles and global bound
            if not self.cycle_detection.add_edge(*comps_ind, test=True):

                rhs = self.global_bound_queue.get_min()[0]

                if abs(path_cost - rhs)<1e-9 or path_cost<=rhs: 

                    # update cycle detection algorithm
                    self.cycle_detection.add_edge(*comps_ind)

                    # after a merge in the separate cycle detection routine
                    # update destinations and reprioritze the open sets
                    if cfg.Algorithm.reprioritize_after_merge:
                            findset = self.cycle_detection.parent_table[comps_ind[0]]
                            # new_goals = {i: self.terminals[i] for i in set(range(len(self.terminals)))-set(findset)}
                            for c in findset:
                                # make sure this set is a completely new object each time!
                                new_goals = {i: self.terminals[i] for i in set(range(len(self.terminals)))-set(findset)}
                                self.comps[(c,)].goal = new_goals
                                self.comps[(c,)].reprioritize()

                    sol.append({'path_cost': path_cost, 'comps_ind': comps_ind})
                    # pop 
                    self.path_queue.get()
                else:
                    break
            else:
                self.path_queue.get()

        return sol