"""Use multiprocessing to find shortest distances"""
# import multiprocessing as mp
import ray.util.multiprocessing as mp
import math
from tkinter import W
import numpy as np
import random
from functools import partial
import logging
from timeit import default_timer as timer

from steinerpy.library.search.search_algorithms import UniSearch, UniSearchMemLimit
from steinerpy.library.search.numba_search_algorithms import UniSearchMemLimitFast
from steinerpy.library.misc.utils import Progress

# DEBUG
import os

my_logger = logging.getLogger(__name__)

class SubPairsShortestPath:
    """Used for computing the compressed differential heuristic (cdh)"""

    @classmethod
    def build(cls, G, node_limit, pivot_limit, subset_limit, processes=4, maxtasksperchild=1000):
        """Given a memory limit build a cdh

        Params:
            node_limit (float or int): Number of surrogate nodes or keys in the lookup table at most |V|
            pivot_limit (int): number of pivots (landmarks) in the graph to use
            subset_limit (int): number of pivot distances per node (surrogate)
        
        That is,
        memory limit = |Pa||K| or subset_limit X node_limit
        """
        # leverage copy-on-write by creating global variables that are shared efficiently
        global graph, pivot_goals
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

        # total_node_count = G.node_count()
        # ind_sz_lim = int(size_limit/pivot_limit)
        # randomly sample free space to get pivots
        pivot_sample = G.sample_uniform(pivot_limit)
        # return a pivot for each pivot
        pivot_identity = {i:p for i,p in enumerate(pivot_sample)}
        # returns the index of a pivot
        pivot_index = {p:i for i,p in enumerate(pivot_sample)}

        # randomly sample surrogate nodes
        surrogate_sample = G.sample_uniform(node_limit)
        # surrogate_sample = set(G.sample_uniform(ind_sz_lim))

        # # assign each surrogate to a pivot in round robin fashion
        # pivot_goals = {}
        # for ndx, s in enumerate(surrogate_sample):
        #     # if ndx%pivot_limit not in pivot_goals:
        #     #     pivot_goals[ndx%pivot_limit] = set()
        #     #     pivot_goals[ndx%pivot_limit].add(s)
        #     # else:
        #     #     pivot_goals[ndx%pivot_limit].add(s)
        #     pivot_goals

        pivot_goals = [[] for _ in range(pivot_limit)]
        skip_by = int(pivot_limit/subset_limit)
        for i, s in enumerate(surrogate_sample):
            for j in range(subset_limit):
                pv_i = int((i+j*pivot_limit/subset_limit) % pivot_limit)
                pivot_goals[pv_i].append(s)

        # pass seed number to guarantee deterministic behavior
        seed_no = range(pivot_limit)

        # construct tasks, each task consists of a pivot, seed number, and goals)
        # tasks = zip(pivot_sample, seed_no, pivot_goals.values())
        tasks = zip(pivot_sample, seed_no) 

        # create paritla function to fix goals
        # task_func = partial(cls._run_dijkstra, surrogate_sample)

        # keep track of job progress
        job_progress = Progress(pivot_limit)

        # save some memory
        del surrogate_sample

        # create multiprocessing pool
        # pool = mp.Pool(processes=processes, maxtasksperchild=maxtasksperchild)
        pool = mp.Pool(ray_address="auto")

        # now run the tasks
        try:
            my_logger.info("Computing cdh tables")
            # for result in pool.imap_unordered(task_func, tasks):
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
        new_data = {"pivot_index": pivot_index, "pivot_identity": pivot_identity, "table": {}}
        
        # table data structure
        struct_dtype = np.dtype([("i" , np.int64), ('d', np.float64)])
        for pivot, values in WORKER_RESULTS.items():
            # if pivot == "type":
            #     new_data[pivot] = values
            #     continue
            for state, dist in values.items():
                # ind = int(abs(pivot_sample.index(pivot) - surrogate_sample.index(state)%pivot_limit)/(pivot_limit/subset_limit))
                # assert ind < subset_limit
                if state not in new_data["table"]:
                    # new_data[state] = {pivot: dist}
                    # consider using sparse matrix here
                    # previous non-sparse way to do things
                    # new_data["table"][state] = np.full(pivot_limit, np.inf)
                    new_data["table"][state] = []
                    # new_data["table"][state] = np.full(pivot_limit, -1, dtype=struct_dtype)

                    # new_data["table"][state] = np.full(subset_limit, np.inf)
                    # last element is the index of the surrogate state ordering
                    # new_data["table"][state] = [float('inf')]*(subset_limit+1)
                    # new_data["table"][state][-1] = surrogate_sample.index
                #     # new_data["table"][state][pivot_index[pivot]] = dist
                # else:
                #     # new_data[state].update({pivot: dist})
                #     new_data["table"][state][pivot_goals[pivot_index[pivot]].index(state)%subset_limit] = (pivot, dist)
                # assert new_data['table'][state][ind] == float('inf') 
                # new_data["table"][state][ind] = dist

                # previous non-sparse way to do things
                # new_data["table"][state][pivot_index[pivot]] = dist
                new_data["table"][state].append((pivot_index[pivot], dist))

        # # convert each value to a structure numpy array
        # for k, v in new_data["table"].items():
        #     new_data["table"][k] = np.array(v, dtype = struct_dtype)

        return new_data
                
    @staticmethod
    def _run_dijkstra(tasks):
        start, seed_no = tasks
        # set seed of numpy
        np.random.seed(seed_no)

        # get some goals (surrogate states)
        # goals = set(graph.sample_uniform(int(ind_sz_lim)))
        goals = set(pivot_goals[seed_no])
        search = UniSearchMemLimitFast(graph, start, goals)

        # print(os.getpid(), start)
        start_time = timer()
        search.use_algorithm()
        # number of nodes expanded
        num_of_expanded = UniSearchMemLimitFast.total_expanded_nodes
        print("nodes expanded", num_of_expanded)
        # time
        total_time = timer() - start_time

        # only get dist to goals
        nvalues = {n:search.g[n] for n in goals}

        # minX, maxX, minY, maxY, minZ, maxZ = graph.grid_dim
        # while len(reduced_dict) < min(total_node_count, ind_sz_lim):
        #    x,y,z = np.random.randint((minX, minY, minZ), (maxX, maxY, maxZ))
        #    if search.g[x,y,z] < np.inf:
        #        reduced_dict[(x,y,z)] = search.g[x,y,z]


        return start, nvalues, num_of_expanded, total_time

class AllPairsShortestPath:
    """Multiple uses:
        - creating the baseline (kruskal)  
        - Building differential heuristic 

    """
    @classmethod
    def dijkstra_in_parallel(cls,G, node_list, processes=4, maxtasksperchild=1000):
        """Find all pairs shortest distance between any pair of vertices in node_list,
        where each vertex belongs to graph G

        Args:
            G (IGraph)
            processes (int)
            maxtasksperchild (int)
            nodes (list of tuples): the vertices for which we want all pairs shortest distance

        Returns:
            results (dict), stats (dict)

        """
        global graph, target_nodes
        graph = G
        target_nodes = node_list

        all_results = {}
        STATS = {"time": 0, "expanded_nodes": 0}

        # # sampling limit
        # limit = kwargs["random_sampling_limit"]
        # node_tasks = G.sample_uniform(limit)
        # num_tasks = len(node_tasks)

        # job_progress = IncrementalBar("Dijkstra in Parallel: ",max=num_tasks)

        job_progress = Progress(len(node_list))

        # pool = mp.Pool(processes=processes, maxtasksperchild=maxtasksperchild)
        pool = mp.Pool(ray_address="auto")
        
        # # flatten dictionary results into a dict of pairs
        # flatten_results = False
        # if "flatten_results_into_pairs" in kwargs:
        #     if kwargs["flatten_results_into_pairs"] == True:
        #         flatten_results = True
                
        try:
            my_logger.info("Running Parallel Dijkstra: ")
            for result in pool.imap_unordered(cls._run_dijkstra, node_list):
                all_results[result[0]] = result[1]
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
        #     for key, value in all_results.items()):
        #         for vkey, vval in value.items():
        #             D[(key, vkey)] = vval
        #         del D[key]
        # if return_stats:
        #     return dict(D), STATS
        # else:
        #     return dict(D)
        return all_results, STATS

    @staticmethod
    def _run_dijkstra(start):
        # search = UniSearch(graph, start, None, "zero", False)
        search = UniSearchMemLimitFast(graph, start, set(target_nodes))
        # print(os.getpid(), start)
        start_time = timer()
        search.use_algorithm()
        # number of nodes expanded
        num_of_expanded = UniSearchMemLimitFast.total_expanded_nodes
        # time
        total_time = timer() - start_time
        # only get dist to target_nodes
        nvalues = {n:search.g[n] for n in target_nodes}
        # return start, search.g, num_of_expanded, total_time
        return start, nvalues, num_of_expanded, total_time

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
