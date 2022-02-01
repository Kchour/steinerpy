"""This module will pre-process graphs loaded in memory to obtain offline heuristics. These results
will be saved to disk and loaded as needed.

For smaller graphs (approx <= 700 nodes), use either floyd-warshall or parallel-dijkstra
to compute all pairs shortest path.

For larger graphs (i.e > 700 nodes), we will have to resort to using "landmarks", which involve
a running limited number of Dijkstra search. Doing so will give a set of "lower bounds".

"""
import numba as nb
import logging
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from steinerpy.library.animation.animationV2 import AnimateV2
from steinerpy.library.graphs.graph import IGraph
from steinerpy.library.misc.utils import Progress

from steinerpy.library.pipeline.base import AFileHandle
from steinerpy.library.search.search_algorithms import UniSearch
import steinerpy.config as cfg

my_logger = logging.getLogger(__name__)

# Voxel grid cardinal length at most 1
vl = 1
SQ3 = 1.7320508075688772
SQ2 = 1.4142135623730951
C1 = SQ3 - SQ2
C2 = SQ2 - vl
C3 = vl

class HFileHandle(AFileHandle):
    """Heuristic file handler object
    
        Default params:
            save_path (str): path to save results
            file_behavior (str): HALT, SKIP, OVERWRITE, RENAME
            load_from_disk (bool):  applicable during "skip"

        Methods:
            _load(): optional function during skip 
            _generate(): generate results 

        We assume the non-expanded heuristic table
        can fit in memory

    """

    def _generate(self):
        results = self.get_heuristics(self.graph, self.processes)
        if ".sqlite" in self.save_path or ".redis" in self.save_path:
            # use sqlitedict to write to disk

            # expand heuristic table
            if results["type"] == "LAND" and self.convert_to_apsp:
                resuls = self.convert_to_apsp(data=results, use_db=True, db_path=self.save_path)
            else:
                # TODO external memory db doesn't seem to work very well
                pass

        elif ".pkl" in self.save_path:
            # use pickle to write to disk

            # expand heuristic table
            if results["type"] == "LAND" and self.convert_to_apsp:
                results = self.convert_to_apsp(data=results, output=self.save_path)
            else:
                with open(self.save_path, 'wb') as f:
                    pickle.dump(results, f)
        
        else:
            raise ValueError("File extension not specified, must either be '.sqlite' or '.pkl' ")

        return results

###################################
# for use with numba
###################################
def numba_dict(ndim):
    """Assumes keys are n-tuples, values are float64
    
    """
    return nb.typed.Dict.empty(
        key_type=nb.types.UniTuple(nb.types.int64, ndim),
        value_type=nb.types.float64,
    )

def numba_nested_dict(ndim):
    return nb.typed.Dict.empty(
        key_type=nb.types.UniTuple(nb.types.int64, ndim),
        value_type=nb.typeof(numba_dict(ndim)),
    )

class GenerateHeuristics:
    #user preloaded
    preload_results = None
    # name or path to heuristic preprocessed database (unused atm)
    preload_name = None 
    # heuristic_type
    preload_type = None

    # for cdh use
    cdh_lower_bound = {}
    cdh_upper_bound = {}
    # lower_bound = None
    # upper_bound = None
    # dist_ap = None

    @classmethod
    def get_heuristics(cls, graph, processes):
        """Entry point for generating heuristics"""
        # get number of nodes
        # n = len(list(graph.get_nodes()))
        n = graph.node_count()

        # STOP THE USER IF THERE ARE TOO MANY NODES
        if n >= 400:
            # use compressed differential heuristic (landmark)
            my_logger.info("Creating COMPRESSED differential heuristic")
            return cls.gen_compressed_diff_heuristic(graph)
        elif n < 400:
            # use differential heuristic (landmark)
            my_logger.info("Creating differential heuristic")

            return cls.gen_landmark_heuristic(graph, processes=processes)
        # else:
        #     # find all pairs shortest distance
        #     my_logger.info("Computing ALL PAIRS SHORTEST DISTANCE")
        #     return cls.gen_all_pairs_shortest_dist(graph, processes=processes)

    @staticmethod
    def gen_compressed_diff_heuristic(graph):
        """wrapper to call cdh builder"""

        # limit size of cdh table to be sqrt(|V|)
        # nodes = len(list(graph.get_nodes()))
        # nodes = graph.node_count()
        # # size_limit = int(math.sqrt(nodes))
        # # size_limit = int(0.50*nodes)

        # # for grid_2d sc
        # # size_limit = 16*nodes
        # # pivot_limit = 100

        # # for grid_3d (simple map)
        # size_limit = nodes/16
        # pivot_limit = 4


        # # for grid_3d (7e6 map)
        # size_limit = nodes/16
        # pivot_limit = 4
        # # r = 

        node_limit = cfg.Pipeline.node_limit
        pivot_limit = cfg.Pipeline.pivot_limit

        from steinerpy.library.search.all_pairs_shortest_path import SubPairsShortestPath

        results = SubPairsShortestPath.build(graph, node_limit, pivot_limit)
        results["type"] = "CDH"
        return results


    @staticmethod
    def gen_landmark_heuristic(graph, processes=4, maxtasksperchild=1000, random_sampling_limit=100):
        """Randomly sample vertices from the graph and give all distances with respect
            to these vertices
        
        results are a dict like object, where each key is mapped to nested dictionary of other nodes and values

        """
        from steinerpy.library.search.all_pairs_shortest_path import AllPairsShortestPath

        results = AllPairsShortestPath.dijkstra_in_parallel(graph, random_sampling_limit=random_sampling_limit)  
        results["type"] = "LAND" 
        return results

    @staticmethod
    def gen_all_pairs_shortest_dist(graph, processes=4, maxtasksperchild=1000):
        """Get all pairs shortest distance from the provided graph

        All possible pair of nodes from the graph are given as keys in the dictionary result

        """
        from steinerpy.library.search.all_pairs_shortest_path import AllPairsShortestPath

        results = AllPairsShortestPath.dijkstra_in_parallel(graph, processes=processes, maxtasksperchild=maxtasksperchild, flatten_results_into_pairs=True)
        results["type"] = "APSP"
        return results

    ################################################################################################
    #   Create lookup table of functions, to mimic switch case speed
    ################################################################################################
    @nb.njit
    def _cdh(cdh_table, ub, lb, from_node, to_node):
        
        if len(from_node)==2:
            x1, y1 = from_node
            x2, y2 = to_node
            dmax = max(abs(x1 - x2), abs(y1 - y2))
            dmin = min(abs(x1 - x2), abs(y1 - y2))
            h1 = 1.414*dmin + (dmax - dmin)
        else:
            x1, y1, z1 = from_node
            x2, y2, z2 = to_node
            dx, dy, dz = abs(x1 - x2), abs(y1 - y2), abs(z1 - z2)
            dmax = max(dx, dy, dz)
            dmin = min(dx, dy, dz)
            dmid = dx + dy + dz - dmin - dmax 
            h1 = C1*dmin + C2*dmid + C3*dmax        

        if from_node not in cdh_table:
            return h1
        else:
            # cdhp = float("-inf")
            cdhp = -math.inf
            # loop over pivots reachable from surrogate "from_node"
            for ndx, (p, dist) in enumerate(cdh_table[from_node].items()):
                a = dist - ub[to_node][p]
                b = lb[to_node][p] - dist
                if a >= b:
                    if a > cdhp:
                        cdhp = a
                else:
                    if b > cdhp:
                        cdhp = b

            if cdhp >= h1:
                return cdhp
            else:
                return h1

    def retrieve_from_cdh(result, from_node, to_node):

        return GenerateHeuristics._cdh(result, 
                                        GenerateHeuristics.cdh_upper_bound, 
                                        GenerateHeuristics.cdh_lower_bound,
                                        from_node,
                                        to_node)

    def retrieve_from_cdh_old(result, from_node, to_node):

        # compute octile heuristic
        x1, y1 = from_node
        x2, y2 = to_node
        dmax = max(abs(x1 - x2), abs(y1 - y2))
        dmin = min(abs(x1 - x2), abs(y1 - y2))
        h1 = 1.414*dmin + (dmax - dmin)

        if from_node not in result:
            return h1
        else:
            # try to use numpy array for speed?
            # dist_ap = np.array(result[from_node].values())
            # p_list = result[from_node].keys()
            # dist_ap = np.fromiter(result[from_node].values(), dtype=np.float32)
            # lower_bound = np.fromiter((GenerateHeuristics.cdh_lower_bound[to_node][p] for p in p_list), dtype=np.float32)
            # upper_bound = np.fromiter((GenerateHeuristics.cdh_upper_bound[to_node][p] for p in p_list), dtype=np.float32)

            cdhp = float("-inf")
            for ndx, (p, dist) in enumerate(result[from_node].items()):
                # a = dist - GenerateHeuristics.cdh_upper_bound[to_node][p]
                # b = GenerateHeuristics.cdh_lower_bound[to_node][p] - dist
                # if a >= b:
                #     if a > cdhp:
                #         cdhp = a
                # else:
                #     if b > cdhp:
                #         cdhp = b

                GenerateHeuristics.dist_ap[ndx] = dist
                GenerateHeuristics.lower_bound[ndx] = GenerateHeuristics.cdh_lower_bound[to_node][p]
                GenerateHeuristics.upper_bound[ndx] = GenerateHeuristics.cdh_upper_bound[to_node][p]
            cdhp = max(np.nanmax(GenerateHeuristics.dist_ap - GenerateHeuristics.upper_bound), 
                                np.nanmax(GenerateHeuristics.lower_bound - GenerateHeuristics.dist_ap))

            # a = np.nanmax(GenerateHeuristics.dist_ap - GenerateHeuristics.upper_bound)
            # b = np.nanmax(GenerateHeuristics.lower_bound - GenerateHeuristics.dist_ap)
            
            # if a >= b:
            #     cdhp = a
            # else:
            #     cdhp = b

            # cdhp = max(np.max(dist_ap - upper_bound), np.max(lower_bound - dist_ap))
            if cdhp >= h1:
                max_lb = cdhp
            else:
                max_lb = h1
            # max_lb = max(cdhp, h1)

            # reset temp arrays
            # GenerateHeuristics.dist_ap[:] = np.inf
            # GenerateHeuristics.upper_bound[:] = np.inf
            # GenerateHeuristics.lower_bound[:] = 0

            return max_lb
            # loop over pivots, dists reachable by "from_node"
            # max_lb = -float('inf')
            # for p, dist_ap in result[from_node].items():
            #     lower_bound = GenerateHeuristics.cdh_lower_bound[to_node][p]
            #     upper_bound = GenerateHeuristics.cdh_upper_bound[to_node][p]
            #     cdhp = max(dist_ap - upper_bound, lower_bound - dist_ap)
            #     # # either [g][p] is stored or not
            #     # try:
            #     #     # [g][p] stored, retrieve dist
            #     #     dist_gp = result[to_node][p]
            #     #     # now compute lower bound from a to g
            #     #     cdhp = abs(dist_ap - dist_gp)
            #     # except KeyError as e:
            #     #     # if pivot to goal not known, compute bounds on dist_gp
            #     #     lower_bound = GenerateHeuristics.cdh_lower_bound[to_node][p]
            #     #     upper_bound = GenerateHeuristics.cdh_upper_bound[to_node][p]
            #     #     cdhp = max(dist_ap - upper_bound, lower_bound - dist_ap)
            #     # except Exception as e:
            #     #     # catch unexpected error
            #     #     raise e

            #     max_lb = max(cdhp, max_lb)

            # assert abs(cdhp1 - max_lb) < 1e-6

            # max_lb = max(max_lb, h1) 
            # return max_lb

    @classmethod
    def cdh_compute_bounds(cls, graph: IGraph, terminals: list):
        """To use CDH, we must perform a bounding procedure
        for each problem instance at run time.

        To use this function correctly:
            1) load the cdh table into memory via 'load_results" function.
            2) generate a problem instance and then call this function 'cdh_compute_bounds'.


        Requires Numba library as we use numba.typed.Dict instead of built-in python

        """
        cdh_table = GenerateHeuristics.preload_results
        # perform a breadth-first search from each terminal (goal state), until
        # the goal is able to reach all of the pivots through some other surrogate goal/state
        def stopping_critiera(self, cdh_table=None, searched_pivots=None, searched_pivot_dists=None, goal_point=None, pivot_set=None, lb=None, ub=None, pivot_counter=None):

            expanded_node = self.current
            current_g_cost = self.g[self.current]
            
            # see if expanded node has indirect path to a pivot
            if expanded_node in cdh_table:
                for pivot in  cdh_table[expanded_node]:
                    # pivot is reachable through surrogate
                    # searched_pivots.add(pivot)
                    # # store distance to surrogate 
                    # if goal_point not in searched_pivot_dists:
                    #     searched_pivot_dists[expanded_node] = current_g_cost
                    
                    # update bounds wrt to pivot directly here
                    dist_g_to_x = current_g_cost
                    dist_x_to_p = cdh_table[expanded_node][pivot]

                    curr_lb = abs(dist_g_to_x - dist_x_to_p)
                    curr_ub = dist_g_to_x + dist_x_to_p

                    if pivot not in lb[goal_point]:
                        lb[goal_point][pivot] = curr_lb
                    else:
                        lb[goal_point][pivot] = max(lb[goal_point][pivot], curr_lb)

                    if pivot not in ub[goal_point]:
                        ub[goal_point][pivot] = curr_ub
                    else:
                        ub[goal_point][pivot] = min(ub[goal_point][pivot], curr_ub)

                    # decrease pivot counter each time we are able to reach it
                    pivot_counter[pivot] -= 1
                    if pivot_counter[pivot] < 0:
                        pivot_counter[pivot] = 0

            # elif expanded_node in pivot_set:
            #     # expanded node forms a direct path to a pivot!
            #     # this rarely happens i believe
            #     if goal_point not in cdh_table:
            #         cdh_table[goal_point]

            # stop once every pivot is reachable (r=1 from table)
            # return searched_pivots == pivot_set

            # stop after reaching all pivots by r amount of times
            return all(x ==0 for x in pivot_counter.values())


        # get all pivots
        pivot_set = set()
        for state, value in cdh_table.items():
            if state == "type":
                continue
            for pivot in value.keys():
                pivot_set.add(pivot)

        if cfg.Pipeline.debug_vis_bounds:
            # debug graphically
            fig, ax = graph.show_grid()
            # plot pivots
            p = np.array(list(pivot_set))
            plt.scatter(p[:,0], p[:,1])

            # # plot surrogate states
            # test = np.array(list(x for x in cdh_table if x != "type"))
            # plt.scatter(test[:,0], test[:,1], marker="*")

            # # draw edges between pivots and surrogate states
            # lines = []
            # for surr, values in cdh_table.items():
            #     if surr == "type":
            #         continue
            #     for pivot, dist in values.items():
            #         lines.append((surr, pivot))
            # # convert to np
            # lines = np.array(lines)
            # lc = LineCollection(lines)
            # ax.add_collection(lc)

            minX, maxX, minY, maxY = graph.grid_dim
            AnimateV2.init_figure(fig, ax, xlim=(minX, maxX), ylim=(minY,maxY))

        # minimum of reaches to a pivot
        r = cfg.Pipeline.min_reach_pivots

        # now run unisearch from terminal
        lb = GenerateHeuristics.cdh_lower_bound
        ub = GenerateHeuristics.cdh_upper_bound
        for t in terminals:
            searched_pivots = set()
            pivot_counter = {p: r for p in pivot_set}
            goal_point = t
            lb[goal_point] = {}
            ub[goal_point] = {}
            search = UniSearch(graph, t, None, "zero", cfg.Pipeline.debug_vis_bounds, stopping_critiera,
                cdh_table=cdh_table, searched_pivots=searched_pivots,
                goal_point=goal_point, pivot_set=pivot_set, 
                lb=lb, ub=ub, pivot_counter=pivot_counter)
        
            search.use_algorithm()

            # compute bounds
            # lb = {}
            # ub = {}
            # # loop over distance to surrogates
            # for surrogate, dist_t_to_s in searched_pivot_dists.items():
            #     pivot, dist_p_to_s = next(iter(cdh_table[(surrogate)].items()))

            #     lb[pivot] = abs(dist_t_to_s - dist_p_to_s) 
            #     ub[pivot] = dist_t_to_s + dist_p_to_s 

            # GenerateHeuristics.cdh_lower_bound[goal_point] = lb
            # GenerateHeuristics.cdh_upper_bound[goal_point] = ub
        # plt.close(fig)
    
        # finish setting up upper and lower bound temp class variables
        # GenerateHeuristics.upper_bound = np.full(len(pivot_set), np.inf, dtype=np.float32)
        # GenerateHeuristics.lower_bound = np.zeros(len(pivot_set), dtype=np.float32)
        # GenerateHeuristics.dist_ap = np.full(len(pivot_set), np.inf, dtype=np.float32)

        # convert cdh relevant dicts into special numba dict types
        one_pivot = next(iter(pivot_set))
        _lb = numba_nested_dict(len(one_pivot))
        _ub = numba_nested_dict(len(one_pivot))
        _cdh = numba_nested_dict((len(one_pivot)))
        for k,values in GenerateHeuristics.cdh_lower_bound.items():
            temp = numba_dict(len(one_pivot))
            for k2, v in values.items():
                temp[k2] = v
            _lb[k] = temp

        for k,values in GenerateHeuristics.cdh_upper_bound.items():
            temp = numba_dict(len(one_pivot))
            for k2, v in values.items():
                temp[k2] = v
            _ub[k] = temp
        
        for k,values in GenerateHeuristics.preload_results.items():
            if k == "type":
                continue
            temp = numba_dict(len(one_pivot))
            for k2, v in values.items():
                temp[k2] = v
            _cdh[k] = temp
        
        # replace loaded stuff with numba types
        GenerateHeuristics.cdh_lower_bound = _lb
        GenerateHeuristics.cdh_upper_bound = _ub
        GenerateHeuristics.preload_results = _cdh

        if cfg.Pipeline.debug_vis_bounds:
            AnimateV2.close()

    def retrieve_from_landmark(result, from_node, to_node):
        """For each landmark, distances to every other node is stored

            Lower bound is computed using triangle inequality
        
            WARNING: This is a slow method
        """

        # compute octile heuristic
        x1, y1 = from_node
        x2, y2 = to_node
        dmax = max(abs(x1 - x2), abs(y1 - y2))
        dmin = min(abs(x1 - x2), abs(y1 - y2))
        h2 = 1.414*dmin + (dmax - dmin)

        # compute heuristic from landmarks (pivots)
        max_lower_bound = max((abs(result[r][from_node] - result[r][to_node]) for r in result if r != "type"))
        return max(max_lower_bound, h2)

    def retrieve_from_apsp(result, from_node, to_node):
        return result[(from_node, to_node)]


    def retrieve_from_db(result, from_node, to_node):
        """access key-value pairs from an a database object (sqlitedict, redis, etc...)
        
        The object can only store string keys, while values 
        can be any pickable object

        """
        if from_node == to_node:
            return 0
        return result[str((from_node, to_node))]


    # binding functions to avoid multiple if statements for the following function
    return_type = {"LAND": retrieve_from_landmark,
                   "CDH" : retrieve_from_cdh, 
                   "APSP": retrieve_from_apsp,
                   "APSP_FROM_LAND": retrieve_from_apsp,
                   "APSP_FROM_DB": retrieve_from_db}
    
    @staticmethod
    def retrieve_heuristic_value(result, from_node, to_node):
        """Entry-point into heuristic look-up table 

        """
        # return GenerateHeuristics.return_type[result["type"]](result, from_node, to_node)
        return GenerateHeuristics.return_type[GenerateHeuristics.preload_type](result, from_node, to_node)

    #########################
    # User interface functions
    #########################
    # @staticmethod
    # def dict_set_wrap(target_dict_obj, temp_dict_obj, use_db):
    #     if use_db:
    #         # dict_like_obj[str(key)] = value
    #         target_dict_obj.update(temp_dict_obj)
    #     else:
    #         dict_like_obj[key] = value
    
    @staticmethod
    def convert_land_to_apsp(filepath=None, data=None, output=None, use_db=False, db_path=None):
        """Get all pairs lower bound values. Either a database or pickle file will be created

        We assume all graphs can be stored in memory

        Params:
            filename (str): location to a preprocessed pickle file with nested dicts
            data (str): if filename is not specified, then the user must pass in the pickle file 
            output (str): optional, location to store result 
            use_db (bool): whether to store a sqliteDict db file or a pickled dict
        
        NOTE: This function is only applicable to grid-based graphs for now

        """
        print("Converting landmarks to All pairs lower bounds (APSP)")
        import pickle
        import itertools as it

        if filepath is not None:
            # if file path is given, load it with pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

        landmarks = data.keys()

        # get all the graph's nodes from the first landmark
        for k in data.keys():
            all_keys = data[k].keys()
            break

        # init sqlite db or dictionary 
        if use_db:
            assert db_path is not None

            from sqlitedict import SqliteDict
            # processed_data =  SqliteDict(db_path, autocommit=True)
            processed_data =  SqliteDict(db_path)

            # from redis_dict import RedisDict
            # processed_data = RedisDict(namespace=db_path)

            processed_data = {}
            # Add type key
            processed_data["type"] = "APSP_FROM_DB"
        else:
            processed_data = {}
            # Add type key
            processed_data["type"] = "APSP_FROM_LAND"

        # now get all pairs max lower bound
        temp = {}
        batch_size = 1e6
        job_progress = Progress(len(list(it.permutations(all_keys, 2))))
        for ndx, (i,j) in enumerate(it.permutations(all_keys, 2)):
            # ignore self edges for now
            if i==j:
                continue

            if len(i) == 2:
                # Octile distance for 2d grid (8 neighbors)
                x1, y1 = i
                x2, y2 = j
                dmax = max(abs(x1 - x2), abs(y1 - y2))
                dmin = min(abs(x1 - x2), abs(y1 - y2))
                h2 = 1.414*dmin + (dmax - dmin)
            elif len(i) == 3:
                # voxel heuristic (octile distance generalized to 3d) (26 neighbors)
                x1, y1, z1 = i
                x2, y2, z2 = j
                dx, dy, dz = abs(x1 - x2), abs(y1 - y2), abs(z1 - z2)
                dmax = max(dx, dy, dz)
                dmin = min(dx, dy, dz)
                dmid = dx + dy + dz - dmin - dmax 
                h2 = C1*dmin + C2*dmid + C3*dmax

            # landmark heuristic
            h1 =  max([abs(data[l][i]-data[l][j]) for l in landmarks if l != "type"]) 

            # max over heuristic lower bounds
            if use_db:
                temp[str((i,j))] = round(max(h1,h2), 5)
            else:
                temp[(i,j)] = round(max(h1,h2), 5)

            # update target dict in batches (which could be sqlite or in memory)
            if ndx % batch_size == 0 and ndx > 0:
                processed_data.update(temp)
                temp = {}

            # debug
            # for l in landmarks: print(l, len(data[l]))

            # GenerateHeuristics.dict_set_wrap(processed_data, (i,j), max(h1,h2), use_db)
            # processed_data[(i,j)] = max(h1,h2) 

            job_progress.next()
        job_progress.finish()

        # make sure to store the last bit of temp 
        processed_data.update(temp)

        if use_db:
            # commit sqlite data
            processed_data.commit()

        # assume self edges to be 0
        # # Add self edges
        # for v in all_keys:
        #     # processed_data[(v,v)] = 0
        #     GenerateHeuristics.dict_set_wrap(processed_data, (v,v), 0, use_db)

        # write pickle file to output path
        if output is not None:
            with open(output, 'wb') as f:
                pickle.dump(processed_data, f)
        
        # return lookup table (either as a )
        return processed_data

    @staticmethod
    def heuristic_wrap(from_node, to_node):
        """wrapper for retrieving heuristic_value
        
        Returns: 
            Float value
        """
        return GenerateHeuristics.retrieve_heuristic_value(GenerateHeuristics.preload_results, from_node, to_node)
        
    @classmethod
    def load_heuristic_name(cls, load_name):
        """Either pass a namespace for redis, a path for
        .sqlite, or a path for .pkl

        """
        cls.preload_name = load_name

    @classmethod
    def load_results(cls, load_location=None, results=None, db_location=None):
        """Interface method to load results into this class's preload cache"""
        if results is not None:
            # user directly gives preprocessed results into memory
            cls.preload_results = results
        elif results is not None:
            # load pickle from disk
            import pickle
            with open(load_location, 'rb') as f:
                results = pickle.load(f)
            cls.preload_results = results
        elif db_location is not None:
            # Load sqliteDict database results from disk
            from sqlitedict import SqliteDict
            results = SqliteDict(db_location, flag='r', journal_mode="OFF")

            # from redis_dict import RedisDict
            # results = RedisDict(namespace=db_location)

            cls.preload_results = results
        else:
            raise ValueError("Nothing action was given")

    @classmethod
    def gen_and_save_results(cls, graph, save_path, processes=4, file_behavior="HALT", convert_to_apsp=False):

        # create a file handle
        fh = HFileHandle(save_path, file_behavior)
        fh.convert_to_apsp = convert_to_apsp
        fh.graph = graph
        fh.processes = processes
        # fh.convert_to_apsp = cls.convert_land_to_apsp
        fh.get_heuristics = GenerateHeuristics.get_heuristics

        # do we need to close .sqlite dict?

        # run 
        return fh.run()


    @classmethod
    def gen_and_save_results_old(cls, graph, file_location, file_name, processes=4, file_behavior=None):
        """Entry point for most users
        
        Returns results if heuristics is generated
        """
        import pickle, os
        
        # location to save generated results
        save_location = os.path.join(file_location, file_name)
        
        # Raise fileExistsError by default 
        if file_behavior is None:
            if os.path.exists(save_location):
                raise FileExistsError('{} already exists!'.format(save_location))
            else:
                # Actually generate the heuristics
                results = cls.get_heuristics(graph, processes)

                # if file path does not exist, create and then save it!
                if not os.path.exists(file_location):
                    os.makedirs(file_location)
                with open(save_location, 'wb') as f:
                    pickle.dump(results, f)

                return results
        else:
            if file_behavior == "SKIP":
                # If the file exists already, just do nothing!
                if os.path.exists(save_location):
                    print("file {} already exists, skipping...".format(file_name) )
                else:
                    results = cls.get_heuristics(graph, processes)
                    # create directory if does not exist
                    if not os.path.exists(file_location):
                        os.makedirs(file_location)
                    with open(save_location, 'wb') as f:
                        pickle.dump(results, f)
                    
                    return results
            elif file_behavior == "OVERWRITE":
                # Does not care if file exists, will overwrite!

                results = cls.get_heuristics(graph, processes)
                # create directory if does not exist
                if not os.path.exists(file_location):
                    os.makedirs(file_location)
                with open(save_location, 'wb') as f:
                    pickle.dump(results, f)

                return results
            elif file_behavior == "RETURNONLY":
                # Does save results, will only return results

                results = cls.get_heuristics(graph, processes)
                return results

            


   