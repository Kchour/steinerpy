import itertools as it
from timeit import default_timer as timer
import numpy as np
import pickle
import logging

import steinerpy.config as cfg
from steinerpy.library.misc.utils import MyTimer
from steinerpy.abstract_algo import AbstractAlgorithm
from steinerpy.library.search.search_utils import PriorityQueueHeap, CycleDetection, reconstruct_path
from steinerpy.library.search.search_algorithms import AStarSearch
from steinerpy.library.animation import AnimateV2
from steinerpy.library.search.all_pairs_shortest_path import AllPairsShortestPath 

my_logger = logging.getLogger(__name__)

class Kruskal(AbstractAlgorithm):

    def __init__(self, G, T):
        super().__init__(G,T)
        # self.terminals = T
        # self.graph = G
        # self.S = {'sol':[], 'dist':[], 'path':[]}

        # For pair-wise path calc
        self.adjQueue = PriorityQueueHeap()

        # For solution
        self.proc = {}      # proc path
        # self.S = {'sol':[], 'dist':[], 'path':[]}       # solution + dist
        indices = [(i,) for i in range(len(self.terminals))]
        self.detector = CycleDetection(indices)

        # # TODO Need to use regex
        # self.graphType = None
        # if 'SquareGrid' in str(type(self.graph)):
        #     self.graphType = "SquareGrid"
        # elif 'SquareGridDepot' in str(type(self.graph)):
        #     self.graphType = "SquareGridDepot"
        # elif 'MyGraph' in str(type(self.graph)):
        #     self.graphType = "MyGraph"

    # def graph_search(self, start, end, heuristic="diagonal_nonuniform"):
    #     """Runs graph search to find shortest path between points 
        
    #     Returns:
    #         parents (dict): A linked list tracing goal back to start node
    #         g (dict): A nodes with a finite g-value

    #     """
    #     i,j = start, end
    #     # # Change heuristics based on underlying graph
    #     # if self.graphType == "SquareGrid":
    #     #     search = AStarSearch(self.graph, i, j, 'diagonal_nonuniform', False) #(works)    
    #     # elif self.graphType == 'SquareGridDepot':
    #     #     search = AStarSearch(self.graph, i, j, 'diagonal_nonuniform', False) #(works)    
    #     # elif self.graphType == "MyGraph":
    #     #     search = AStarSearch(self.graph, i, j, 'zero', False)
    #     # parents, g = search.use_algorithm()

    #     search = AStarSearch(self.graph, i, j, heuristic, False)
    #     parents, g = search.use_algorithm()

    #     return parents, g
    
    def run_algorithm(self, heuristic_for_recon='diagonal_nonuniform', RECON_PATH=True):

        # TODO: If cache file is provided, we can skip parallel dijkstra and just look precomputed values

        # do all-pairs-shortest path between terminals via parallel dijkstra
        apsp_terminals, pd_stats = AllPairsShortestPath.dijkstra_in_parallel(self.graph, nodes=self.terminals)
        self.S['stats']['expanded_nodes'] = pd_stats['expanded_nodes']
        self.S['stats']['time'] = pd_stats['time']

        # construct a graph-like dict from the terminals
        metric_graph_edge_dict = {}
        for edge in it.combinations(self.terminals,2):
            metric_graph_edge_dict[edge] = apsp_terminals[edge[0]][edge[1]]

        # sort the edges in the graph-like dict
        sorted_edges = sorted(metric_graph_edge_dict, key=lambda x: metric_graph_edge_dict[x])

        # now produce mst 
        try:
            if self.graph.graph_type == "undirected":
                # then make sure we don't double count edges in undirected
                for edge in sorted_edges[0::2]:
                    # append to solution
                    self.S['sol'].append(edge)
                    self.S['dist'].append(self.graph.cost(*edge))
            elif self.graph.graph_type == "directed":
                for edge in sorted_edges:
                    # append to solution
                    self.S['sol'].append(edge)
                    self.S['dist'].append(self.graph.cost(*edge))
            else:
                raise ValueError("graph type is not specified")
        except:
            my_logger.error("graph_type must either be directed or undirected", exc_info=True)
        
        # # TODO A better Visualization process
        # if cfg.Animation.visualize:
        #     if self.graphType == "SquareGrid" or "SquareGridDepot":
        #         minX, maxX, minY, maxY = self.graph.grid_dim
        #         # grid_size = self.graph.grid_size

        #         # plotTerminals = Animate(number=1, xlim=(minX, maxX), ylim=(minY, maxY), gridSize=grid_size, linewidth=5, markerType='ko', markerSize=20, sleep=0.5, order=2)
        #         # plotPath = Animate(number=1, xlim=(minX, maxX), ylim=(minY, maxY), gridSize=grid_size, linewidth=5, markerType='ro', markerSize=5, sleep=0.5, order=3)
        #         # plotTerminals.update(np.array(self.terminals).T.tolist())
        #         AnimateV2.add_line("terminals", np.array(self.terminals).T.tolist(), xlim=(minX, maxX), ylim=(minY,maxY), markersize=14, marker="x", color='y',zorder=12)

        #         # if any
        #         if self.graph.obstacles:
        #             # plotObstacle = Animate(number=1, xlim=(minX, maxX), ylim=(minY, maxY), gridSize=grid_size, linewidth=5, markerType='o', markerSize=2, sleep=0.5, order=2)
        #             # plotObstacle.update(np.array(self.graph.obstacles).T.tolist())
        #             AnimateV2.add_line("obstacles", np.array(self.graph.obstacles).T.tolist(), markersize=5, marker='o', color='k',zorder=11)

        #         if self.graphType == "SquareGridDepot" and self.graph.depots:
        #             AnimateV2.add_line("depots", np.array(self.graph.depots).T.tolist(), markersize=14, marker='o', color='b',zorder=12)

        #         AnimateV2.update()            

        # # Start time
        # startLoop = timer()

        # # print("Running Kruskals")
        # if file_to_read is None:
        #     temp_expanded = []
        #     for t in self.terminals:
        #         parents, g = self.graph_search(start=t, end=set(self.terminals)-set({t}), heuristic='zero')
        #         temp_expanded.append(AStarSearch.total_expanded_nodes)
        #         for t2 in self.terminals:
        #             if t2 != t:
        #                 path = reconstruct_path(parents, t, t2)
        #                 self.proc.update({(t,t2): path})
        #                 priority = g[t2]
        #                 self.adjQueue.put((t,t2), priority)


        #     # for i,j in it.combinations(self.terminals,r=2):            
                
        #     #     parents, g = self.graph_search(start=i, end=j,  heuristic=desired_heuristic)
        #     #     # Get priority value
        #     #     priority = g[j]

        #     #     # Store path for reuse?
        #     #     path = reconstruct_path(parents, i, j)
        #     #     self.proc.update({(i,j): path})
        #     #     self.adjQueue.put((i,j), priority)
                
        #     #     # if cfg.visualize:
        #     #     #     if self.graphType == "SquareGrid" or self.graphType == "SquareGridDepot":
        #     #     #         AnimateV2.update()       
        # else:
        #     # build AllPairsShortestPath from cache(s)    
        #     temp_proc, temp_time, temp_expanded = [], [], []
        #     cache_values = {}
        #     for ff in file_to_read:
        #     # read from file(s), load into adjQueue
        #         with open(ff, 'rb') as f:
        #             cache_ = pickle.load(f)    

        #         for i,j in it.combinations(self.terminals, r=2):
        #             # get terminal pair stats from cache (it should be)
        #             if (i,j) in cache_:
        #                 time_, dist_, expanded_ = cache_[(i,j)]['time'], cache_[(i,j)]['dist'], cache_[(i,j)]['expanded']                       
        #                 self.adjQueue.put((i,j),  dist_) 
        #                 cache_values.update({(i,j): {'time':time_}})
        #                 cache_values[(i,j)].update({'expanded':expanded_})
        #                 # temp_time.append(cache_[(i,j)]['time'])
        #                 # temp_expanded.append(cache_[(i,j)]['expanded'])                   
        #             if (j,i) in cache_:
        #                 time_, dist_, expanded_ = cache_[(j,i)]['time'], cache_[(j,i)]['dist'], cache_[(j,i)]['expanded']
        #                 self.adjQueue.put((i,j),  dist_) 
        #                 cache_values.update({(j,i): {'time':time_}})
        #                 cache_values[(j,i)].update({'expanded':expanded_})
        #                 # temp_time.append(cache_[(j,i)]['time'])
        #                 # temp_expanded.append(cache_[(j,i)]['expanded'])   
               
        #     if len(cache_values) == 0:
        #         raise ValueError("why is extracted cache empty???")
            
        # sol_nodes_expanded, sol_time_expanded = [],[]
        # while not self.adjQueue.empty():
        #     # Pop priorityQueue 
        #     dist, t = self.adjQueue.get()
        #     # Whether to read from file
        #     if file_to_read is None:
        #         path = self.proc[t]

        #     # index of terminal pair returned
        #     idx1 = (self.terminals.index(t[0]), )
        #     idx2 = (self.terminals.index(t[1]), )
        #     # Check for cycles
        #     status = self.detector.add_edge(idx1, idx2)
        #     if status is not True:
        #         # print("cost: ", dist)
        #         self.S['sol'].append((t[0], t[1]))
        #         self.S['dist'].append(dist)
        #         if file_to_read is None:
        #             self.S['path'].append(path)
        #         else:
        #             temp_proc.append(t)
        #             temp_time.append(cache_values[t]['time'])
        #             temp_expanded.append(cache_values[t]['expanded'])           

        #     # Early break using disjoint set algorithm size
        #     if len(self.detector.indices) is 1:
        #         break
        
        # if len(self.S['dist']) == 0:
        #     raise ValueError("Why is dist array empty???")

        # # End time
        # endLoop = timer()

        # # if reading from file, add paths posthumously
        # if  file_to_read is not None:
        #     if RECON_PATH:
        #         for t in temp_proc:
        #             parents, _ = self.graph_search(start=t[0], end=t[1], heuristic=heuristic_for_recon)
        #             path = reconstruct_path(parents, t[0], t[1])
        #             self.S['path'].append(path)

        # # Plot paths
        # for path in self.S['path']:
        #     if cfg.Animation.visualize and 'SquareGrid' == self.graphType or 'SquareGridDepot' == self.graphType:
        #         # plotPath.update(np.array(path).T.tolist())
        #         AnimateV2.add_line("solution", np.array(path).T.tolist(), 'ro', markersize=10, zorder=10)
        #         AnimateV2.update()
            
        # # print length
        # totalLen = sum(np.array(self.S['dist']))
        # # print("finished: ", totalLen)

        # #########################
        # ###     ADD STATS     ###
        # #########################

        # if file_to_read is None:
        #     self.S['stats']['expanded_nodes'] = sum(temp_expanded)
        #     self.S['stats']['time'] = endLoop - startLoop
        # else:
        #     self.S['stats']['expanded_nodes'] = sum(temp_expanded)
        #     self.S['stats']['time'] = endLoop - startLoop + sum(temp_time)


        # # RESET Class variables (find a better way to to do this)
        # AStarSearch.reset()

        # # TODO A better visualization process
        # if cfg.Animation.visualize:
        #     import matplotlib.pyplot as plt
        #     plt.show()

        # # # TO BREAK LOOP
        # # self.FLAG_STATUS_completeTree = True
        # return True

    def return_solutions(self):
        return self.S

# class KruskalMulti(Kruskal):

#     def __init__(self, G, T):
#         super().__init__(G,T)

#     def run_algorithm(self):
        

    
