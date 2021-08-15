"""This module will pre-process graphs loaded in memory to obtain offline heuristics. These results
will be saved to disk and loaded as needed.

For smaller graphs (approx <= 700 nodes), use either floyd-warshall or parallel-dijkstra
to compute all pairs shortest path.

For larger graphs (i.e > 700 nodes), we will have to resort to using "landmarks", which involve
a running limited number of Dijkstra search. Doing so will give a set of "lower bounds".

"""
import logging

my_logger = logging.getLogger(__name__)

class GenerateHeuristics:
    #user preloaded
    preload_results = None

    @classmethod
    def get_heuristics(cls, graph, processes):
        # get number of nodes
        n = len(list(graph.get_nodes()))

        # STOP THE USER IF THERE ARE TOO MANY NODES
        if n > 800:
            # use landmark heuristic method
            my_logger.info("GENERATING LANDMARKS")

            return cls.gen_landmark_heuristic(graph, processes=processes)
        else:
            # find all pairs shortest distance
            my_logger.info("Computing ALL PAIRS SHORTEST DISTANCE")
            return cls.gen_all_pairs_shortest_dist(graph, processes=processes)

    @staticmethod
    def gen_landmark_heuristic(graph, processes=4, maxtasksperchild=1000, random_sampling_limit=100):
        """Randomly sample vertices from the graph and give all distances with respect
            to these vertices
        
        results are a dictionary, where each key is mapped to nested dictionary of other nodes and values

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

    @staticmethod
    def retrieve_from_landmark(result, from_node, to_node):
        """For each landmark, distances to every other node is stored

            Lower bound is computed using triangle inequality
        
            WARNING: This is a slow method
        """
        max_lower_bound = max((abs(result[r][from_node] - result[r][to_node]) for r in result if r != "type"))
        return max_lower_bound 

    def retrieve_from_apsp(result, from_node, to_node):
        return result[(from_node, to_node)]

    return_type = {"LAND": retrieve_from_landmark, 
                   "APSP": retrieve_from_apsp,
                   "APSP_FROM_LAND": retrieve_from_apsp}
    @staticmethod
    def retrieve_heuristic_value(result, from_node, to_node):
        """Non-databased version of heuristic value retrieval 

        TODO implement database with sqllite and APSW?


        """
        return GenerateHeuristics.return_type[result["type"]](result, from_node, to_node)

    #########################
    # User interface functions
    #########################

    @staticmethod
    def convert_land_to_apsp(filename=None, data=None, output=None):
        """User can either specify the file location or provide the data from memory

        """
        print("Converting landmarks to All pairs shortest path (APSP)")
        import pickle
        import itertools as it
        
        if filename is not None:

            with open(filename, 'rb') as f:
                data = pickle.load(f)


        # get landmarks
        landmarks = data.keys()

        # get all the dijkstra keys
        for k in data.keys():
            all_keys = data[k].keys()
            break

        # get all pairs and put into dictionary
        processed_data = {}
        for (i,j) in it.permutations(all_keys, 2):
            # Octile distance
            x1, y1 = i
            x2, y2 = j
            dmax = max(abs(x1 - x2), abs(y1 - y2))
            dmin = min(abs(x1 - x2), abs(y1 - y2))
            h2 = 1.414*dmin + (dmax - dmin)

            # landmark heuristic
            h1 =  max([abs(data[l][i]-data[l][j]) for l in landmarks if l != "type"]) 
        
            # max over heuristic lower bounds
            processed_data[(i,j)] = max(h1,h2) 

        # Add self edges
        for v in all_keys:
            processed_data[(v,v)] = 0

        # Add type key
        processed_data["type"] = "APSP_FROM_LAND"

        if output is not None:
            with open(output, 'wb') as f:
                pickle.dump(processed_data, f)
        
        return processed_data

    @staticmethod
    def heuristic_wrap(from_node, to_node):
        """wrapper for the above method 
        
        Returns: 
            Float value
        """
        return GenerateHeuristics.retrieve_heuristic_value(GenerateHeuristics.preload_results, from_node, to_node)
        
    @classmethod
    def load_results(cls, load_location=None, results=None):
        """Interface method to load results from disk into memory """
        if results is not None:
            cls.preload_results = results
        else:
            import pickle
            with open(load_location, 'rb') as f:
                results = pickle.load(f)
            cls.preload_results = results

    @classmethod
    def gen_and_save_results(cls, graph, file_location, file_name, processes=4, file_behavior=None):
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

            


   