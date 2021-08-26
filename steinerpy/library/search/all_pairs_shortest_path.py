import multiprocessing as mp
import math
import numpy as np
from functools import partial
import logging
from timeit import default_timer as timer

from steinerpy.library.search.search_algorithms import AStarSearch
from steinerpy.library.misc.utils import Progress

# DEBUG
import os

my_logger = logging.getLogger(__name__)

class AllPairsShortestPath:

    @classmethod
    def dijkstra_in_parallel(cls,G, processes=4, maxtasksperchild=1000, flatten_results=False, return_stats=False, **kwargs):
        """Solve APSP problem with running Dijkstra on each node. Alternatively,
        the user can specify a subset of nodes or a random percentage of all nodes
        to consider

        Args:
            G (IGraph)
            processes (int)
            maxtasksperchild (int)
            random_sampling_percentage (int): optional
            nodes (list of IGraph nodes): optional
            random_sampling_limit (int): optional

        Returns:
            results (dict), stats (dict)

        """
        global graph
        graph = G

        D = {}
        STATS = {"time": 0, "expanded_nodes": 0}


        # Run Dijkstra based on sampling technique:
        # 1) limit samples to a percentage of the configuration space.
        # 2) fixed samples given by user.
        # 3) limit samples to a fixed number of the configuration space.
        if "random_sampling_percentage" in kwargs:
            # len_ = G.node_count()
            node_tasks = []
            ## Look at boundary nodes only
            # all_nodes = list(G.get_nodes())
            all_nodes = list(G.get_boundary_nodes())
            limit = kwargs["random_sampling_percentage"]/100.*len(all_nodes)

            while len(node_tasks) < limit:
                node_tasks.append(all_nodes[np.random.randint(len(all_nodes)-1)])
            # clean up
            del all_nodes, limit
            num_tasks = len(node_tasks)
            pass
        elif "nodes" in kwargs:
            node_tasks = kwargs["nodes"]  
            num_tasks = len(node_tasks)      
            pass
        elif "random_sampling_limit" in kwargs:
            limit = kwargs["random_sampling_limit"]
            node_tasks = []
            ## Look at boundary nodes only
            # all_nodes = list(G.get_nodes())
            all_nodes = list(G.get_boundary_nodes())
            while len(node_tasks) < limit:
                node_tasks.append(all_nodes[np.random.randint(len(all_nodes)-1)])
            # clean up
            del all_nodes
            num_tasks = len(node_tasks) 
            pass
        else:
            # default without any arguments
            num_tasks = G.node_count()
            node_tasks = G.get_nodes()

        # job_progress = IncrementalBar("Dijkstra in Parallel: ",max=num_tasks)
        job_progress = Progress(num_tasks)

        pool = mp.Pool(processes=processes, maxtasksperchild=maxtasksperchild)
        
        # flatten dictionary results into a dict of pairs
        flatten_results = False
        if "flatten_results_into_pairs" in kwargs:
            if kwargs["flatten_results_into_pairs"] == True:
                flatten_results = True
                
        try:
            my_logger.info("Running Parallel Dijkstra: ")
            for result in pool.imap_unordered(cls._run_dijkstra, node_tasks):
                D[result[0]] = result[1]
                STATS["expanded_nodes"] += result[2]
                STATS["time"] += result[3]
                job_progress.next()
                pass
        except Exception as e:
            pool.terminate()
            pool.close()
            pool.join()
            raise e
        
        job_progress.finish()
        pool.close()
        pool.join()    
        if flatten_results:
            for key, value in list(D.items()):
                for vkey, vval in value.items():
                    D[(key, vkey)] = vval
                del D[key]
        if return_stats:
            return dict(D), STATS
        else:
            return dict(D)

    @staticmethod
    def _run_dijkstra(start):
        search = AStarSearch(graph, start, None, "zero", False)
        # print(os.getpid(), start)
        start_time = timer()
        search.use_algorithm()
        # number of nodes expanded
        num_of_expanded = AStarSearch.total_expanded_nodes
        # time
        total_time = timer() - start_time
        return start, search.g, num_of_expanded, total_time

    @classmethod
    def floyd_warshall_simple_slow(cls, G):
        """Floyd Warshall Algorithm implemented without 
            any changes to entering data structure
        Args:
            G (IGraph): Graph instance which subclasses IGraph
        Returns:
            D (dict): A table of all pairs, keyed by shortest distance
        """
        D = G.get_adjacency_matrix()
        n = D.shape[0]
        # progress_bar = IncrementalBar("fw sequential", max=n)
        progress_bar = Progress(n)

        for k in range(n):
            progress_bar.next()
            for i in range(n):
                for j in range(n):
                    try:
                        D[i][j] = min(D[i][j], D[i][k] + D[k][j])
                    except Exception as e_:
                        raise e_
        progress_bar.finish()
        # return dictionary of distances, adjacency to dictionary
        D_to_dict={}
        for i in range(n):
            for j in range(n):
                D_to_dict.update({(G.adj_map_to_wc[j], G.adj_map_to_wc[i]): D[j][i]})

        return D_to_dict

    @classmethod
    def floyd_warshall_simple_slow_parallel(cls, graph, processes=None, maxtasksperchild=1000):
        # Create manager for sharing memory between processes
        # manager = mp.Manager()
        # D = manager.dict()

        # number of processes to use
        if processes is None:
            processes = math.floor(mp.cpu_count)/2
                
        global G
        G = graph

        # Get edges from graph
        D = graph.get_adjacency_matrix()
        n = D.shape[0]

        # create pool
        pool = mp.Pool(processes=processes, maxtasksperchild=maxtasksperchild)      

        # Create inner-loop tasks
        # cls.G = G

        # Total progress
        # progress_bar = IncrementalBar("fw in parallel", max=n)
        progress_bar = Progress(n)

        # generate results k times. Add progress bar
        for k in range(n):
            # create partial target function
            f = partial(cls.inner_loop_slow_parallel, k, D, n)
            progress_bar.next()

            # # debug
            # f(task = ((5, 5), (5, 5)))
            try: 
                for results in pool.imap_unordered(f, range(n), chunksize=int(n//processes**2+1)):
                    D[results[0]] = results[1]
            except Exception as e_:
                # early termination due to exception
                pool.terminate()
                pool.close()
                pool.join()
                raise e_

        # Good practice, ensure no zombies
        pool.close()
        pool.join()
        progress_bar.finish()
        
        # return dictionary of distances, adjacency to dictionary
        D_to_dict={}
        for i in range(n):
            for j in range(n):
                D_to_dict.update({(G.adj_map_to_wc[j], G.adj_map_to_wc[i]): D[j][i]})

        return D_to_dict

    @staticmethod
    def inner_loop_slow_parallel(k, D, n, task):
        # global G
        i = task
        # print("Acquiring lock...")
        # l.acquire()
        # print("Lock acquired!", i,j)   
        try:
            for j in range(n):
                D[i][j] = min(D[i][j], D[i][k] + D[k][j])
        except Exception as e_:
            raise e_
        return i, D[i]