"""This module is used to produce baseline files """

import pickle
import random as rd
import os
from timeit import default_timer as timer
import multiprocessing as mp
import math
import numpy as np
import itertools as it
from functools import partial
import logging

from steinerpy.library.search.all_pairs_shortest_path import AllPairsShortestPath
from steinerpy.library.search.search_algorithms import AStarSearch
# from steinerpy.library.logger import MyLogger
from steinerpy.context import Context
import steinerpy.config as cfg
from steinerpy.library.misc.utils import Progress
# from steinerpy.algorithms.kruskal import Kruskal

my_logger = logging.getLogger(__name__)

class GenerateBaseLine:
    """Generate a baseline file using Kruskal's algorithm, to help compare with S* algorithm

    The user has the option of setting the number of terminals to be randomly generated and 
    number of run-instances. The required output file name is the form 'baseline_{}t-{}i.pkl'

    Attributes:
        graph (SquareGrid, MyGraph): Graph classes from superclass IGraph.
            Created using 'GraphFactory' class from the 'graph' module
        n_terminals (int): Number of terminals for a given instance
        m_instances (int): Number of run instances or trials
        filename (str): The output name for our baseline file
        save_directory (str): The absolute path of output files
        minX (float): the minimum X coordinate for our 'SquareGrid' graph. Also similarily
            defined for maxX, minY, maxY
    Example: 
        >>> # Spec out our squareGrid
        >>> minX = -15			# [m]
        >>> maxX = 15           
        >>> minY = -15
        >>> maxY = 15
        >>> grid = None         # pre-existing 2d numpy array?
        >>> grid_size = 1       # grid fineness[m]
        >>> grid_dim = [minX, maxX, minY, maxY]
        >>> n_type = 8           # neighbor type
        >>> ... 
        >>> # Create a squareGrid using GraphFactory
        >>> from steinerpy.library.graphs.graph import GraphFactory
        >>> graph = GraphFactory.create_graph("SquareGrid", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type)      
        >>> ...
        >>> # generate m instances with N terminals
        >>> N = 10
        >>> instances = 10
        >>> #create and run
        >>> gb = GenerateBaseLine(graph, N, instances, 'baseline_{}t-{}i.pkl'.format(N,instances))
        >>> gb.run_func()

    Todo: 
        * Add support for general 'MyGraph' types
        
    """
    heuristic_type = "diagonal_nonuniform"
    reconstruct_path_ornot = True

    def __init__(self, graph, n_terminals, m_instances, save_directory, filename='baseline.pkl', save_to_file=True):
        
        self.graph = graph
        self.n_terminals = n_terminals
        self.m_instances = m_instances
        self.filename = filename
        self.save_directory = save_directory

        self.minX, self.maxX, self.minY, self.maxY = graph.grid_dim

        # getter setter for terminals
        self._terminals = None

        # whether to save file
        self.save_to_file = save_to_file

        # Detect landlocked regions using dijkstra
        free_locs = np.where(self.graph.grid==0)
    
    @property
    def terminals(self):
        """ Accessor method for terminals """
        return self._terminals

    @terminals.setter
    def terminals(self, terminals):
        """Setter method for terminal property """
        self._terminals = terminals

    def create_terminals(self, read_from_file=None):
        """ Create at most N, unique terminals

        Returns:
            list_of_instances (list of set of tuples): A list of list of terminal tuples
       
        Todo:
            * Make sure generated point is not landlocked

        """
        # array = set()
        # while len(array) < self.n_terminals:
        #     pt = (rd.randint(self.minX, self.maxX), rd.randint(self.minY, self.maxY))
        #     # array.add(pt)
        #     # Make sure terminal doesn't coincide with an obstacle if any
        #     if self.graph.obstacles is not None:
        #         if pt not in self.graph.obstacles:
        #             array.add(pt)
        #     else:
        #         array.add(pt)
          
        # yield array
        if read_from_file is None:
            list_of_instances = []
            for _ in range(self.m_instances):
                terminal_set = set()
                while len(terminal_set) < self.n_terminals:
                    # randomly generate a point
                    pt = (rd.randint(self.minX, self.maxX), rd.randint(self.minY, self.maxY))
                    # make sure point is unique using set and not an obstacle!
                    if self.graph.obstacles is not None :
                        if  pt not in self.graph.obstacles:
                            terminal_set.add(pt)
                    else:
                        terminal_set.add(pt)
                list_of_instances.append(list(terminal_set))

        else:
            with open(read_from_file, 'rb') as f:
                list_of_instances = pickle.load(f)
        
        return list_of_instances
                        
    def run_func(self):
        """ Queries Kruskal's algorithm for each run instance for baseline results

        Results are saved to file

        Raises:
            FileExistsError: If output (baseline) file already exists
        
        Todo:
            * Consider using timeit for more accurate results
                https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python

        """
        #start time
        t0 = timer()

        # Get path and check for existance. Don't overwrite files!
        # directory = os.path.dirname(os.path.realpath(__file__))
        if os.path.exists(os.path.join(self.save_directory, self.filename)):
            raise FileExistsError('{} already exists!'.format(self.filename))

        # Create n-terminals for m-instances 
        if self._terminals is None:
            # self._terminals = [[list(i) for i in self.create_terminals()][0] for j in range(self.m_instances)]
            self._terminals = self.create_terminals()
        obstacles = self.graph.obstacles

        # Run Kruskals on each and save results
        solution = []
        for ndx, t in enumerate(self._terminals):
            my_logger.info("Running terminals: {}".format(t))

            # ko = Kruskal(self.graph, t)
            # ko.run_algorithm()
            # solution.append(ko.return_solutions())
            # Ensure every terminal is reachable!
            keepRunning = True
            while keepRunning:
                try:
                    context = Context(self.graph, t)
                    context.run('Kruskal')
                    solution.append(context.return_solutions())
                    keepRunning = False
                except:
                    my_logger.error("one or more terminals is landlocked! In ndx {}, with terminals: {}".format(ndx,t))
                    # Generate new terminals and replace the ones we have right now
                    t = [list(i) for i in self.create_terminals()][0]
                    self._terminals[ndx] = t

            print(ndx, '\n')

        # end time
        t1 = timer() - t0   

        # dump instances and solution to file
        # directory = os.getcwd()
        if self.save_to_file:
            with open(os.path.join(self.save_directory, self.filename), 'wb') as f:
                pickle.dump({
                    'terminals': self._terminals,
                    'solution': solution,
                    'obstacles': obstacles
                }, f)

        print("Finished! Wrote baseline file! Now generate results! {}".format(t1))
        if cfg.Misc.sound_alert == True:
            os.system('spd-say "Finished! Wrote baseline file! Now generate results!"')

        return {
                'terminals': self._terminals,
                'solution': solution,
                'obstacles': obstacles
               }

class GenerateBaseLineMulti(GenerateBaseLine):
    """Generate baseline using multiple cores 
    
    Attributes:
        file_behavior (str): OVERWRITE-overwrites an existing file, SKIP-skip over an existing file,
            HALT-stop when encountering an existing file, RENAME-append number to new file when file exists,
            LOAD-load previous terminal data

    Todo:
        * How to handle errors in child processes when certain areas of a map are landlocked?

    """
    def __init__(self, graph, n_terminals, m_instances, save_directory, filename, cache_filename=None, save_to_file=True, num_processes=4, file_behavior="OVERWRITE"):
        super().__init__(graph, n_terminals, m_instances, save_directory, filename=filename, save_to_file=save_to_file)
        # FIXME Make sure we try to detect physical cores only; Exclude logical ones
        self.num_processes = num_processes
        
        # Create an on-disk cache to store previous single source shortest distances
        # to save time from recalculating
        # 
        # Data format is a dictionary as follows:
        # {((x1,y1), (x2,y2)): 
        #       {'dist': float_value,
        #        'expanded': int_value,
        #        'time': float_value}
        # }
        # Stored as a binary file through pickle (FIXME: change to cPickle)
        if cache_filename is None:
            self.cache_filepath = os.path.join(self.save_directory, "__cache__" + filename)
        else:
            self.cache_filepath = os.path.join(self.save_directory, cache_filename)

        # Append number to cache if it exists
        cnt = 0
        temp = self.cache_filepath
        self.cachefiles = []
        while True:
            if os.path.exists(temp + str(cnt).zfill(3)):
                self.cachefiles.append(temp+str(cnt).zfill(3))
                cnt += 1
            else:
                break
        self.cache_count = cnt

        # Temporary file to store randomly generated terminals. Gives us the option to reload between runs of this module
        self.temp_terminals_filename = os.path.join(self.save_directory, "__temp__"+filename)
        
        self.file_behavior_list = ["OVERWRITE", "SKIP", "HALT", "RENAME", "LOAD"]
        if file_behavior not in self.file_behavior_list:
            raise ValueError("Action {} not defined. Must be of the following: {}".format(file_behavior, self.file_behavior_list))
        else:
            self.file_behavior = file_behavior

        self.filename = filename

    def init(self,l):
        """Required to avoid race conditions for lists

        Info:
            https://stackoverflow.com/questions/25557686/python-sharing-a-lock-between-processes

        """
        global lock
        lock = l
        

    def run_func(self):
        #start time
        # t0 = time.time()os.path.join(self.save_directory, "__cache__"+self.filename)

        # Get path and check for existance. Don't overwrite files!
        # directory = os.path.dirname(os.path.realpath(__file__))
        # TODO put this in a context manager class
        SKIP = False	#
        LOAD = True
        if self.file_behavior == "HALT":
            if os.path.exists(self.temp_terminals_filename):
                raise FileExistsError('{} already exists!'.format(self.temp_terminals_filename))
        elif self.file_behavior == "OVERWRITE":
            pass
        elif self.file_behavior == "SKIP":
            if os.path.exists(os.path.join(self.save_directory, self.filename)):
                SKIP=True
        elif self.file_behavior == "RENAME":
            cnt = 1
            while True:
                temp = self.filename
                if os.path.exists(os.path.join(self.save_directory, temp)):
                    temp += str(cnt)
                    cnt += 1
                else:
                    self.filename = temp
                    break
        elif self.file_behavior == "LOAD":
            LOAD = True

        len_terminal_set = 0 
        if SKIP is not True:
            # CREATE N-TERMINALS FOR M-INSTANCES 
            if self._terminals is None:
                if LOAD == False:
                    self._terminals = self.create_terminals()
                    if self.file_behavior == "OVERWRITE":
                        with open(self.temp_terminals_filename, 'wb') as f:
                            pickle.dump(self._terminals, f) 
                else:
                    # Check to see if temp even exists for RESUSE. If not definitely generate and save to disk
                    if not os.path.exists(self.temp_terminals_filename):
                        self._terminals = self.create_terminals()
                        with open(self.temp_terminals_filename, 'wb') as f:
                            pickle.dump(self._terminals, f) 
                    else:
                        self._terminals = self.create_terminals(self.temp_terminals_filename)
                        self.m_instances = len(self._terminals)
                        self.n_terminals = len(self._terminals[0])

            # obstacles = self.graph.obstacles

            # Coalesce terminal instances into one set
            terminal_set = set()
            for i in range(self.m_instances):
                for j in range(self.n_terminals):
                    terminal_set.add(self._terminals[i][j])
            len_terminal_set = len(terminal_set) 

            # Reuse data from cache if it exists, so don't have to look at all combination of paths
            # first get all combinations from terminal set
            allcombs_term_set = set(it.combinations(terminal_set, 2))
            for cf in self.cachefiles:
                if os.path.exists(cf):
                    with open(cf, 'rb') as f:
                        results_from_file = pickle.load(f)
                                       
                    # subtract solved problems from results_from_File
                    allcombs_term_set -= set(results_from_file)

                    # len_terminal_set  = len(list((x  for x in set(it.combinations(terminal_set, 2)) - set(results_from_file))))
                    # terminal_set = set(x  for x in it.combinations(terminal_set, 2)) - set(results_from_file)
                    
                    # #create a node set from results from file
                    # results_from_file_set = set()
                    # for v in results_from_file.keys():
                    #     results_from_file_set.add(v[0])
                    #     results_from_file_set.add(v[1])
                    
                    # terminal_set = terminal_set - results_from_file_set
                    # len_terminal_set = len(terminal_set)
                    
                    if len_terminal_set == 0:
                        break
                    del results_from_file
            # turn all pairs back into single set of terminals
            terminal_set = set()
            for v in allcombs_term_set:
                terminal_set.add(v[0])
                terminal_set.add(v[1])
            
            del allcombs_term_set
            len_terminal_set = len(terminal_set)
 
        # Multiprocess to get all (subset) pairs shortest path). Dont run this section if no tasks OR if baseline already exists
        if len_terminal_set > 0:
            # temporarily save our terminals to disk to relieve memory usage
            if LOAD == False:
                with open(self.temp_terminals_filename, 'wb') as f:
                    pickle.dump(self._terminals, f)
            # clear _terminals attribute
            self._terminals = None
            self.terminal_set = terminal_set
            
            ### Rest of the code is for multi-astar search over terminal_set (m*n choose 2)
            
            # Create a lock, must be done before the pool
            l = mp.Lock()
            
            # # Create manager for shared object
            # manager = mp.Manager()
            # path_results_dict = manager.dict()
            
            # limit number of processes to use
            if self.num_processes is None:
                self.num_processes = math.floor(mp.cpu_count()/2)
            # create pool of limited number of processes to run
            pool = mp.Pool(processes=self.num_processes, maxtasksperchild=1000)

            # use partial function for fixed objects
            # partFunc = partial(self.get_path_info, self.graph, path_results_dict)
            
            # Now iterate over pool, add a progress bar and timer
            t0 = timer()

            #use try/finally clause to ensure pool closure
            try:
                bar_assign_job = Progress(len_terminal_set)
                chunksize = int(len_terminal_set // (self.num_processes**2) + 1)
                print("Tentative jobs: ", len_terminal_set)
                print("Chunksize: ",chunksize)# update the cache if it exists. if not create one
                
                partial_results = {}
                for pd in pool.imap_unordered(self.get_path_info, terminal_set, chunksize):
                    # # path_results_dict[pd[0]] = pd[1]
                    # path_results_dict.update(pd)
                    
                    partial_results.update(pd)

                    # save results to disk if len is too great
                    if len(partial_results) > 300000:
                        # save results to disk
                        with open(self.cache_filepath + str(self.cache_count).zfill(3), 'wb') as f:
                            pickle.dump(partial_results, f)
                        
                        # update cache files direct path and count
                        self.cachefiles.append(self.cache_filepath + str(self.cache_count).zfill(3))
                        self.cache_count += 1

                        # reset partial results
                        partial_results = {}

                    bar_assign_job.next()
                
                # Make sure to save the last chunk of data if non-empty
                if len(partial_results)>0:
                    # save results to disk
                    with open(self.cache_filepath + str(self.cache_count).zfill(3), 'wb') as f:
                        pickle.dump(partial_results, f)
                    
                    self.cachefiles.append(self.cache_filepath + str(self.cache_count).zfill(3))
                    partial_results={}

                bar_assign_job.finish()
                t1 = timer()-t0
            except Exception as e_:
                pool.terminate()
                raise e_
            finally:
                # good practice
                pool.close()
                pool.join()

            # # update the cache if it exists. if not create one
            # if os.path.exists(self.cache_filepath):
            #     with open(self.cache_filepath, 'rb') as f:
            #         results_from_file = pickle.load(f)
            #         # update with our new data
            #         results_from_file.update(dict(path_results_dict))
            #     with open(self.cache_filepath, 'wb') as f:
            #         pickle.dump(results_from_file, f)
            # else:
            #     with open(self.cache_filepath, 'wb') as f:
            #         pickle.dump(dict(path_results_dict), f)
        
        # AUTOSKIPS IF FILE ALREADY EXISTS
        if not SKIP:
            # Reload terminal instances post mp
            with open(os.path.join(self.temp_terminals_filename), 'rb') as f:
                self._terminals = pickle.load(f)
            ### Call Kruskal for each terminal instance
            # # Create a lock, must be done before the pool
            # l = mp.Lock()

            # Create manager for shared object
            # manager = mp.Manager()
            # kruskal_results_proxy= manager.list()
            kruskal_results = []

            # limit number of processes to use
            if self.num_processes is None:
                self.num_processes = math.floor(mp.cpu_count()/2)
            pool = mp.Pool(processes=self.num_processes, maxtasksperchild=1000)

            # use partial function for fixed objects
            # partFunc = partial(self.get_kruskal_info, kruskal_results_proxy)
            t0 = timer()
            # add try except finally
            try:
                bar_assign_job = Progress(len(self._terminals))
                for kd in pool.imap(self.get_kruskal_info, self._terminals, int(len(self._terminals) // (self.num_processes**2) + 1)):
                    kruskal_results.append(kd)
                    bar_assign_job.next()
                bar_assign_job.finish()
                t1 = timer()-t0
            except Exception as e_:
                pool.terminate()
                raise e_
            finally:
                # good practice
                pool.close()
                pool.join()

            # solution = []
            # for t in self._terminals:
            #     context = Context(self.graph, t)
            #     context.run('Kruskal', file_to_read=self.cache_filename)
            #     solution.append(context.return_solutions())

            # Write baseline results to file
            if self.save_to_file:
                # Create folder if it doesn't exist already
                if not os.path.exists(self.save_directory):
                    os.makedirs(self.save_directory)
                with open(os.path.join(self.save_directory, self.filename), 'wb') as f:
                    pickle.dump({
                        'terminals': self._terminals,
                        'solution': kruskal_results,
                        'obstacles': self.graph.obstacles
                    }, f)

            print("Finished! Wrote baseline file! Now generate results! {}".format(t1))
            if cfg.Misc.sound_alert == True:
                os.system('spd-say "Finished! Wrote baseline file! Now generate results!"')

            return {
                    'terminals': self._terminals,
                    'solution': kruskal_results,
                    'obstacles': self.graph.obstacles
                }

    def get_kruskal_info(self, terminal_instance):
        context = Context(self.graph, terminal_instance)
        # context.run('Kruskal', file_to_read=self.cache_filepath, heuristic_for_recon=self.heuristic_type, RECON_PATH=self.reconstruct_path_ornot)
        context.run('Kruskal', file_to_read=self.cachefiles, heuristic_for_recon=self.heuristic_type, RECON_PATH=self.reconstruct_path_ornot)
        # lock.acquire()
        # kruskal_results_proxy.append(context.return_solutions())
        # lock.release()
        return context.return_solutions()

    # define target function for pool
    def get_path_info(self, nodes):
        
        # Older stuff, which wassss working
        # # print("Current PID", os.getpid(), "performing get_path_info(nodes)...")
        # start, end = nodes
        # # Create and use Astar search class
        # search = AStarSearch(self.graph, start, end, self.heuristic_type, False)
        
        # TRY RUNNING DIJKSTRA IN PARALLEL INSTEAD
        start = nodes
        # search = AStarSearch(self.graph, start, self.terminal_set, 'zero', False)
        search = AStarSearch(self.graph, start, None, 'zero', False)


        t0 = timer()
        _, g = search.use_algorithm()
        t1 = timer() - t0
        # # get opt_dist
        # opt_dist = g[end]

        # # lock.acquire()
        # # proxy_dict[(start, end)] = {"dist": opt_dist, "expanded": search.total_expanded_nodes, "time": t1}
        # # lock.release()
        # # print("Current PID", os.getpid(), "FINISHED!!!")
        # return (start, end), {"dist": opt_dist, "expanded": search.total_expanded_nodes, "time": t1}

        # construct dictionary to return
        subpairs_dict = {(start, end): {"dist": g[end], "expanded": search.total_expanded_nodes, "time": t1} for end in self.terminal_set if end != start}
        search.reset()
        return subpairs_dict 

if __name__ == "__main__":
    # Spec out our squareGrid
    minX = -15			# [m]
    maxX = 15           
    minY = -15
    maxY = 15
    grid = None         # pre-existing 2d numpy array?
    grid_size = 1       # grid fineness[m]
    grid_dim = [minX, maxX, minY, maxY]
    n_type = 8           # neighbor type
    # Create a squareGrid using GraphFactory
    from steinerpy.library.graphs.graph import GraphFactory
    graph = GraphFactory.create_graph("SquareGrid", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type)      

    # generate m instances with N terminals
    N = 2
    instances = 10

    import steinerpy.config as cfg
    save_directory = cfg.results_dir + "/tests"
    # specify directory to write baseline file to
    # directory = os.path.dirname(os.path.realpath(__file__))+"/../"

    #create and run
    gb = GenerateBaseLine(graph, N, instances, save_directory, 'baseline_{}t-{}i.pkl'.format(N,instances))
    gb.run_func()
