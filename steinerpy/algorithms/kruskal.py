import itertools as it
from timeit import default_timer as timer
import numpy as np
import pickle
import logging

import steinerpy.config as cfg
from steinerpy.library.misc.utils import MyTimer
from steinerpy.abstract_algo import AbstractAlgorithm
from steinerpy.library.search.search_utils import PriorityQueueHeap, CycleDetection, reconstruct_path
from steinerpy.library.animation import AnimateV2
from steinerpy.library.search.all_pairs_shortest_path import AllPairsShortestPath 

my_logger = logging.getLogger(__name__)

class Kruskal(AbstractAlgorithm):

    def __init__(self, G, T):
        super().__init__(G,T)
        # self.terminals = T
        # self.graph = G
        # self.results = {'sol':[], 'dist':[], 'path':[]}

        # For pair-wise path calc
        self.adjQueue = PriorityQueueHeap()

        # For solution
        self.proc = {}      # proc path
        # self.results = {'sol':[], 'dist':[], 'path':[]}       # solution + dist
        indices = [(i,) for i in range(len(self.terminals))]
        self.detector = CycleDetection(indices)
    
    def run_algorithm(self, heuristic_for_recon='diagonal_nonuniform', RECON_PATH=True):

        # TODO: If cache file is provided, we can skip parallel dijkstra and just look precomputed values

        # do all-pairs-shortest path between terminals via parallel dijkstra
        apsp_terminals, pd_stats = AllPairsShortestPath.dijkstra_in_parallel(self.graph, node_list=self.terminals)
        self.results['stats']['expanded_nodes'] = pd_stats['expanded_nodes']
        self.results['stats']['time'] = pd_stats['time']

        # now produce mst 
        try:
            if self.graph.edge_type == "UNDIRECTED":
                # undirected edges requires looking at all pair combinations (don't double count edges)

                # construct a graph-like dict from the terminals
                metric_graph_edge_dict = {edge: apsp_terminals[edge[0]][edge[1]] for edge in it.combinations(self.terminals,2)}

                # sort the edges in the graph-like dict
                sorted_edges = sorted(metric_graph_edge_dict, key=lambda x: metric_graph_edge_dict[x])

                for edge in sorted_edges:
                     # index of terminal pair returned
                    idx1 = (self.terminals.index(edge[0]), )
                    idx2 = (self.terminals.index(edge[1]), )
                    # Check for cycles
                    if not self.detector.add_edge(idx1, idx2):
                        self.results['sol'].append(edge)
                        self.results['dist'].append(metric_graph_edge_dict[edge])
                        # break after getting a tree
                        if len(self.results['sol']) == len(self.terminals):
                            break
            elif self.graph.edge_type == "DIRECTED":
                # directed edges requires looking at all pair permutations

                # construct a graph-like dict from the terminals
                metric_graph_edge_dict = {edge: apsp_terminals[edge[0]][edge[1]] for edge in it.permutations(self.terminals,2)}

                # sort the edges in the graph-like dict
                sorted_edges = sorted(metric_graph_edge_dict, key=lambda x: metric_graph_edge_dict[x])

                for edge in sorted_edges:
                    # index of terminal pair returned
                    idx1 = (self.terminals.index(edge[0]), )
                    idx2 = (self.terminals.index(edge[1]), )
                    # Check for cycles
                    if not self.detector.add_edge(idx1, idx2):
                        self.results['sol'].append(edge)
                        self.results['dist'].append(metric_graph_edge_dict[edge])
                    # break after getting a tree
                    if len(self.results['sol']) == len(self.terminals):
                        break
            else:
                raise ValueError("graph type is not specified")
        except ValueError as e_ :
            # handle expected error
            my_logger.error("graph_type must either be directed or undirected", exc_info=True)
            raise e_
        except Exception as e_:
            # handle unexpected error
            raise e_
        
        
        return True

    def return_solutions(self):
        return self.results

        

    
