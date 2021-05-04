"""This module will pre-process graphs loaded in memory to obtain offline heuristics. These results
will be saved to disk and loaded as needed.

For smaller graphs (approx <= 700 nodes), use either floyd-warshall or parallel-dijkstra
to compute all pairs shortest path.

For larger graphs (i.e > 700 nodes), we will have to resort to using "landmarks", which involve
a running limited number of Dijkstra search. Doing so will give a set of "lower bounds".

"""
from steinerpy.library.logger import MyLogger

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
            MyLogger.add_message("GENERATING LANDMARKS", __name__, "INFO")

            return cls.gen_landmark_heuristic(graph, processes=processes)
        else:
            # find all pairs shortest distance
            MyLogger.add_message("ALL PAIRS SHORTEST DISTANCE", __name__, "INFO")
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

    @staticmethod
    def retrieve_heuristic_value(result, from_node, to_node):
        """Non-databased version of heuristic value retrieval 

        TODO implement database with sqllite and APSW?

        """
        if result["type"] == "LAND":
            max_lower_bound = max((abs(result[r][from_node] - result[r][to_node]) for r in result if r != "type"))
            return max_lower_bound 
        elif result["type"] == "APSP":
            return result[(from_node, to_node)]

    #########################
    #########################
    #########################


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
        """Entry point for most users"""
        import pickle, os
        
        save_location = os.path.join(file_location, file_name)
        # Raise fileExistsError by default
        if file_behavior is None:
            if os.path.exists(save_location):
                raise FileExistsError('{} already exists!'.format(save_location))
        else:
            if file_behavior == "SKIP":
                if os.path.exists(save_location):
                    pass
                else:
                    results = cls.get_heuristics(graph, processes)
                    # create directory if does not exist
                    if not os.path.exists(file_location):
                        os.makedirs(file_location)
                    with open(save_location, 'wb') as f:
                        pickle.dump(results, f)
            elif file_behavior == "OVERWRITE":
                results = cls.get_heuristics(graph, processes)
                # create directory if does not exist
                if not os.path.exists(file_location):
                    os.makedirs(file_location)
                with open(save_location, 'wb') as f:
                    pickle.dump(results, f)
            elif file_behavior == "RETURNONLY":
                results = cls.get_heuristics(graph, processes)
                return results

            


   