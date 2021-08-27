"""Implementation of an algorithm that builds a connected proximity graph (Gabriel), based on S*-Unmerged

    TODO: Better to not inherit framework, consider creating a different base class

"""
from collections import OrderedDict
import numpy as np
import logging

from steinerpy.library.animation import AnimateV2
from steinerpy.library.search.search_utils import reconstruct_path
from steinerpy.algorithms.common import Common
from steinerpy.library.search.generic_algorithms import GenericSearch
import steinerpy.config as cfg
from steinerpy.library.misc.utils import MyTimer

from .unmerged import Unmerged

my_logger = logging.getLogger(__name__)

class Proximity(Unmerged):

    def __init__(self, graph, terminals):
        super().__init__(graph, terminals)

    def tree_update(self):
        solQueue = self.solution_handler()

        for ndx, s in enumerate(solQueue):
            t1, t2 = s

            dist, commonNode = self.UFeasPath[t1][t2]

            path, _, term_actual = Common.get_path(comps=self.comps, sel_node=commonNode, term_edge=(t1, t2), reconstruct_path_func=reconstruct_path)

            self.add_solution(path=path, dist=dist, edge=term_actual)

            # TODO find a better way to animate path
            if cfg.Animation.visualize:
                # self.animateS.update_clean(np.vstack(self.S['path']).T.tolist())
                AnimateV2.add_line("solution", np.vstack(self.S['path']).T.tolist(), 'yo', markersize=10, zorder=10)

    def tree_check(self):
        my_logger.info("Performing tree_check")

        if cfg.Animation.visualize:
            # Don't plot every thing for large graphs
            if np.mod(self.run_debug, np.ceil(self.graph.edge_count()/5000))==0:
                AnimateV2.update()

        if self.FLAG_STATUS_pathConverged:
           
            # Check tree size
            if len(self.S['sol']) == len(self.terminals)-1:
                # Algorithm has finished
                self.FLAG_STATUS_completeTree = True
                totalLen = sum(np.array(self.S['dist']))

                my_logger.info("Finished: {}".format(totalLen))

                # Add expanded node stats
                self.S['stats']['expanded_nodes'] = GenericSearch.total_expanded_nodes
                # Reset or "close" Class variables

                # Add additional stats (don't forget to reset classes)
                self.S['stats']['fcosts_time'] = sum(MyTimer.timeTable["fcosts_time"])

                # Keep plot opened
                if cfg.Animation.visualize:
                    # ### Redraw closed + frontier regions with fixed color
                    # # k = self.comps.keys()
                    # # self.comps[k[0]].g
                    # # self.comps[k[0]]
                    # # xo = []
                    # # yo = []
                    # # for n in self.comps[k[0]].frontier.elements:
                    # #     xo.append(n[0])
                    # #     yo.append(n[1])
                    # # AnimateV2.add("closed_{}".format(self.comps[k[0]].id), dataClosedSet[0], dataClosedSet[1], 'o', markersize=10, draw_clean=True)
                    # # AnimateV2.add("neighbors_{}".format(self.comps[k[0]].id), xo, yo, 'D', color='c', markersize=10, draw_clean=True)
                    
                    # recolor the final plot so that all open sets, closed sets have fixed color
                    terminal_handle = None
                    for artist, art_dict in AnimateV2.instances[1].artists.items():
                        # Set closed sets to the same color: magenta
                        if "closed" in artist:
                            art_dict['artist'][0].set_markerfacecolor("magenta")
                            art_dict['artist'][0].set_markerfacecoloralt("magenta")
                            art_dict['artist'][0].set_markeredgecolor("magenta")

                        # Set open sets to the same color: cyan
                        if "neighbors" in artist:
                            art_dict['artist'][0].set_markerfacecolor("cyan")
                            art_dict['artist'][0].set_markerfacecoloralt("cyan")
                            art_dict['artist'][0].set_markeredgecolor("cyan")

                        # Set open sets to the same color: cyan
                        if "solution" in artist:
                            art_dict['artist'][0].set_markerfacecolor("yellow")
                            art_dict['artist'][0].set_markerfacecoloralt("yellow")
                            art_dict['artist'][0].set_markeredgecolor("yellow")


                        if "terminal" in artist:
                            terminal_handle = art_dict['artist'][0]
                    AnimateV2.update()

                    import matplotlib.pyplot as plt
                    from matplotlib.patches import Rectangle
                    closed_rect = Rectangle((0, 0), 0, 0 , fc="magenta", fill=True, edgecolor=None, angle=45, linewidth=1)
                    open_rect = Rectangle((0, 0), 0, 0 , fc="cyan", fill=True, edgecolor=None, linewidth=1, angle=45)
                    sol_rect = Rectangle((0, 0), 0, 0 , fc="yellow", fill=True, edgecolor=None, linewidth=1)

                    labels = ["closed-set", "open-set", "tree-path", 'terminals']

                    plt.legend([closed_rect, open_rect, sol_rect, terminal_handle], labels, ncol=4, bbox_to_anchor=(0, 1.10, 1, 0), loc="lower left")
                    # import re
                    # search = re.search(r'.*algorithms.(\w+).(\w+)', str(self))
                    # alg_name = search.group(2)
                    # ax = AnimateV2.instances[1].ax
                    # ax.set_title(alg_name)

                    plt.tight_layout()  #to make legend fit
                    ax = plt.gca()
                    # ax.axis('equal')
                    # ax.set_aspect('equal', 'box')

                    plt.draw()
                    plt.pause(1)

            self.FLAG_STATUS_pathConverged = False

    def solution_handler(self):
        """Get the shortest path between terminals. Avoid duplicates

        """
        sol = dict()
        my_logger.info("Len of path queue: {}".format(len(self.pathQueue.elements)))

        while not self.pathQueue.empty():
            poppedQ = self.pathQueue.get()
            dist, comps_ind = poppedQ

            # make sure we don't have duplicates or reverse edges
            ind1, ind2 = comps_ind
            if (self.terminals[ind2[0]], self.terminals[ind1[0]]) not in self.S['sol'] and (self.terminals[ind1[0]], self.terminals[ind2[0]]) not in self.S['sol'] and \
                (ind2, ind1) not in sol and (ind1, ind2) not in sol:
                sol[comps_ind] = dist
        return sol

    def add_solution(self, dist, path, edge):
        # Add solution if triangle inequality is respected!
        self.S['dist'].append(dist)
        self.S['path'].append(path)
        self.S['sol'].append(edge)
        my_logger.debug("Added edge no.: {}".format(len(self.S['sol']))) 
