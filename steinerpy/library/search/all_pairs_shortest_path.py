from ast import keyword
from concurrent.futures import process
import multiprocessing as mp
import math
import numpy as np
import random
from functools import partial
import logging
from timeit import default_timer as timer
from steinerpy.config import Animation

from steinerpy.library.search.search_algorithms import UniSearch, UniSearchMemLimit
from steinerpy.library.misc.utils import Progress

# DEBUG
import os

my_logger = logging.getLogger(__name__)

class SubPairsShortestPath:
    """Used for computing the compressed differential heuristic (cdh)"""

    @classmethod
    def build(cls, G, size_limit, pivot_limit, processes=4, maxtasksperchild=1000):
        """Given a memory limit build a cdh

        Params:
            size_limit (float or int): The limit in the number of items in the cdh 
            pivot_limit (int): number of pivots (landmarks) in the graph to use
        
        Typically,
        m := |P_a|, the number of pivot distances per state 'a' \in V
        |P| := number of pivots  
        memory_limit = m|V|, m >= 0

        """
        # leverage copy-on-write by creating global variables that are shared efficiently
        global graph, ind_sz_lim, total_node_count
        graph = G

        WORKER_RESULTS = {}
        STATS = {"time": 0, "expanded_nodes":0}

        # # randomly generate pivots
        # all_nodes = list(G.get_nodes())
        # total_node_count = len(all_nodes)
        # tasks = set() 
        # while len(tasks) < pivot_limit:
        #     r_node = random.choice(all_nodes)
        #     tasks.add(r_node)
        # tasks = list(tasks)
        # del all_nodes


        total_node_count = G.node_count()
        ind_sz_lim = int(size_limit/pivot_limit)
        # randomly sample free space to get pivots
        tasks = G.sample_uniform(pivot_limit)

        # keep track of job progress
        job_progress = Progress(len(tasks))

        # create multiprocessing pool
        pool = mp.Pool(processes=processes, maxtasksperchild=maxtasksperchild)

        # now run the tasks
        try:
            my_logger.info("Computing cdh tables")
            for result in pool.imap_unordered(cls._run_dijkstra, tasks):

                # for i in range(total_node_count-ind_sz_lim):
                #     result[1].pop(random.choice(list(result[1].keys())))

                # store individual worker result
                WORKER_RESULTS[result[0]] = result[1]

                # update job progress
                job_progress.next()

        except Exception as e:
            pool.terminate()
            pool.close()
            pool.join()
            raise e
        
        # notify job finished
        job_progress.finish()
        pool.close()
        pool.join()

        # transpose results so that keys are all states, values are {pivot: dist}
        new_data = {}
        for pivot, values in WORKER_RESULTS.items():
            if pivot == "type":
                new_data[pivot] = values
                continue
            for state, dist in values.items():
                if state not in new_data:
                    new_data[state] = {pivot: dist}
                else:
                    new_data[state].update({pivot: dist})

        return new_data
                
    @staticmethod
    def _run_dijkstra(start):
        # this is regular dijkstra
        # search = UniSearch(graph, start, None, "zero", False)
        # debugging purposes
        # from steinerpy.library.animation.animationV2 import AnimateV2
        # fig, ax = graph.show_grid()
        # AnimateV2.init_figure(fig, ax)

        # this is memory limited dijkstra
        search = UniSearchMemLimit(graph, start, None, ind_sz_lim, "zero", False)

        # print(os.getpid(), start)
        start_time = timer()
        search.use_algorithm()
        # number of nodes expanded
        num_of_expanded = UniSearchMemLimit.total_expanded_nodes
        # time
        total_time = timer() - start_time

        # # limit individual items in the results to size_limit/pivot
        # keys_to_del = set()
        # all_keys = list(search.g.keys())
        # while len(keys_to_del)< (total_node_count - ind_sz_lim):
        #     keys_to_del.add(random.choice(all_keys))
        # for k in keys_to_del:
        #     search.g.pop(k)

        return start, search.g, num_of_expanded, total_time

class AllPairsShortestPath:
    """Multiple uses:
        - creating the baseline (kruskal)  
        - Building differential heuristic 

    """
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

        # sampling limit
        limit = kwargs["random_sampling_limit"]
        node_tasks = G.sample_uniform(limit)
        num_tasks = len(node_tasks)

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
        search = UniSearch(graph, start, None, "zero", False)
        # print(os.getpid(), start)
        start_time = timer()
        search.use_algorithm()
        # number of nodes expanded
        num_of_expanded = UniSearch.total_expanded_nodes
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