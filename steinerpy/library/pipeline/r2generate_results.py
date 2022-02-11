"""Generate results using S* algorithms. Don't run Kruskal here"""

import pickle
import os
from timeit import default_timer as timer
import re
import math
import itertools as it
# import multiprocessing as mp
from functools import partial
import logging
from typing import List, Union


import ray.util.multiprocessing as mp

# from results.debug_heuristics import pre_run_func
# from steinerpy.library.pipeline.r0generate_heuristics import GenerateHeuristics

from .base import Generate

from steinerpy.context import Context
import steinerpy.config as cfg
from steinerpy.library.misc.utils import Progress
from steinerpy.library.graphs import IGraph


my_logger = logging.getLogger(__name__)

class GenerateResults(Generate):
    """Generate results based on baseline files, so that we can process the results.

    Note: 
        Prior to running this file, the baseline file must already exist. If not,
        run 'r1generate_baseline.py' or 'r1agenerate_baseline_obstacles.py'.     

    Args:
        graph (SquareGrid, MyGraph): Graph classes from superclass IGraph.
            Created using 'GraphFactory' class from the 'graph' module   
        load_directory (str): Location to load baseline files
        baseline_file (str): Name of baseline file in the previous directory (optional)
        save_directory (str, None): Location to save generated result files. If None,
            defaults to the load_directory
        run_instance (int, None): By default None which generates results for all instances.
            Else, generate result for only a particular instance
        save_to_file (bool): Whether or not to save results to disk

    Attributes:
        graph (SquareGrid, MyGraph): Graph classes from superclass IGraph.
            Created using 'GraphFactory' class from the 'graph' module            
        minX (float): the minimum X coordinate for our 'SquareGrid' graph. Also similarily
            defined for maxX, minY, maxY
        baseline_file (str): Name of the input baseline filename. To be opened
            with pickle. 
        run_instance (int, None): If not None, run a particular instance
        save_to_file (bool): Whether to write to file or not
        load_directory (str): Location of baseline files
        save_directory (str): Location to save generated results
        results_filename (str): Auto created based on baseline_file name
        data (dict): baseline data, which has the following structure
            data={ 
            'solution': [{'sol':[], 'path':[], 'dist':[]}, {...}, ..., {}]
            'terminals':[ [(x,y),...,(xn,yn)],[...],[...],....,[] ]
            'obstacles': [(x,y),...,(xn,yn) ]       # Assumed slightly changing
            }

    Todo: 
        * Currently only supports SquareGrid, need to add support for any general graph
        * Don't allow the user to overwrite generated results.
        * Rename generated results according to baseline file

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
        >>> gr = GenerateResults(graph, 'baseline_10t-11i-50o.pkl')
        >>> gr.run_func()

    """
    def __init__(self, graph: IGraph, save_path: str="", file_behavior: Union["SKIP", "HALT", "RENAME", "OVERWRITE"]="HALT", algs_to_run:List[str]=None, load_from_disk=False):
        
        if algs_to_run is None:
            raise ValueError("No algorithms were specified!")
        elif "Kruskal" in algs_to_run:
            raise ValueError("Cannot run Kruskal here, use r1generate_baseline instead")
        
        self.algs_to_run = algs_to_run

        self.solution = {alg: [] for alg in self.algs_to_run}

        super().__init__(graph, save_path, file_behavior, load_from_disk)

    def _generate(self):
        """Generate results defined by algs_to_run
        
            Returns:
                results (dict): key-value are alg_name-solutions

        """
        #start time
        t0 = timer()
        # keep track of progress
        pg = Progress(len(self.instances)*len(self.algs_to_run))

        for ndx, t in enumerate(self.instances):
            try: 
                my_logger.info("Running (single-core) terminals: {}".format(t))

                context = Context(self.graph, t)

                for alg in self.algs_to_run:
                    context.run(alg)
                    self.solution[alg].append(context.return_solutions())

                    pg.next()
            except Exception as e: 
                my_logger.error("Something terrible has happened", exc_info=True)
                raise e

        pg.finish()

        # end time
        t1 = timer() - t0

        # dump instances and solution to file if path specified
        if self.save_path != "":
            with open(self.save_path, 'wb') as f:
                pickle.dump({
                    'terminals': self.instances,
                    'solution': self.solution,
                }, f)

            print("Wrote results to file! Now process them! {} secs".format(t1))
            if cfg.Misc.sound_alert == True:
                os.system('spd-say "Finished! Wrote results to file! Now process them!"')

        return {
                'terminals': self.instances,
                'solution': self.solution,
               }

class GenerateResultsMulti(Generate):

    # class variables to allow global scope
    graph = None
    pre_run_func = None

    def __init__(self, graph: IGraph, save_path: str="", file_behavior: str="HALT", num_processes=4, maxtasksperchild=50, algs_to_run=None,
                pre_run_func=None, *kwargs):
        
        if algs_to_run is None:
            raise ValueError("No algorithms were specified!")
        elif "Kruskal" in algs_to_run:
            raise ValueError("Cannot run Kruskal here, use r1generate_baseline instead")
        
        self.maxtasksperchild = maxtasksperchild
        self.num_processes = num_processes

        # self.pre_run = pre_run_func
        # self.pre_run_kwargs = kwargs

        GenerateResultsMulti.graph = graph
        GenerateResultsMulti.pre_run_func = pre_run_func

        self.algs_to_run = algs_to_run
        self.solution = {alg: [] for alg in self.algs_to_run}
        super().__init__(graph, save_path, file_behavior)


    def _generate(self):
        #start time
        t0 = timer()
        # keep track of progress
        number_of_jobs = len(self.instances)*len(self.algs_to_run)
        pg = Progress(number_of_jobs)   

        # create jobs for the pool
        jobs = it.product(self.instances, self.algs_to_run)
        # local variable create after pool ensures
        # children processes don't have it
        solution = self.solution.copy()
        # create pool
        # pool = mp.Pool(processes=self.num_processes, maxtasksperchild=self.maxtasksperchild)
        pool = mp.Pool(ray_address="auto", processes=8)

        func = partial(GenerateResultsMulti._run_individual_algs)

        t0 = timer()
        try:
            # for res in pool.imap_unordered(GenerateResultsMulti._run_individual_algs, enumerate(jobs), chunksize=int(number_of_jobs // (self.num_processes**2) + 1)):
            # for res in pool.imap_unordered(GenerateResultsMulti._run_individual_algs, jobs): 
            for res in pool.map(func, jobs): 
                solution[res[0]].append(res[1])
                pg.next()
            pg.finish()
        except Exception as _e:
            pool.terminate()
            my_logger.error("Something has gone wrong with GenerateResultsMulti", exc_info=True)
            print(_e)
        finally:
            # good practice
            pool.close()
            pool.join()

        # end time
        t1 = timer() - t0


        self.solution = solution

        # dump instances and solution to file if path specified
        if self.save_path != "":
            with open(self.save_path, 'wb') as f:
                pickle.dump({
                    'terminals': self.instances,
                    'solution': self.solution,
                }, f)

            print("Wrote results to file! Now process them! {} secs".format(t1))
            if cfg.Misc.sound_alert == True:
                os.system('spd-say "Finished! Wrote results to file! Now process them!"')

        return {
                'terminals': self.instances,
                'solution': self.solution,
               }

    @staticmethod
    def _run_individual_algs(inputs):
        # job_id, data = inputs
        # print("starting job id: ", job_id)

        # grab from class variables
        graph = GenerateResultsMulti.graph
        pre_run_func = GenerateResultsMulti.pre_run_func

        terminals, alg = inputs
        context = Context(graph, terminals)

        # self.terminals = terminals
        # self.alg = alg 
        # self.job_id = job_id

        # heuristic specific pre-run operations
        # from steinerpy.library.pipeline import GenerateHeuristics
        
        # do pre-run operations here (like performing cdh bounding)
        # # load heuristics (have each process connect to database...)
        # if GenerateHeuristics.preload_name is not None:
        #     GenerateHeuristics.load_results(db_location=GenerateHeuristics.preload_name)
        if cfg.Pipeline.perform_prerun_r2 and pre_run_func is not None:
            pre_run_func(graph, terminals)

        context.run(alg)
        # print("finished job id: ", job_id)

        return alg, context.return_solutions()

    # def init(self,l):
    #     """Required to avoid race conditions for lists

    #     Info:
    #         https://stackoverflow.com/questions/25557686/python-sharing-a-lock-between-processes

    #     """
    #     global lock
    #     lock = l

    # def run_func(self, algs_to_run: List[str]): 
    #   if not self.SKIP: 
    #     # Create task generator, progress bar
    #     # algs_to_run = ['SstarAstar', 'SstarPrimalDual', 'SstarMM', "Astar"]
    #     number_of_jobs = len(list(it.product(self.data['terminals'], algs_to_run)))
    #     bar_assign_job = Progress(number_of_jobs)
    #     tasks = it.product(self.data['terminals'], algs_to_run)
    #     # unload data attribute for memory relief
    #     self.data = None

    #     # create Manager for shared object between processes
    #     # https://stackoverflow.com/questions/41780322/why-is-a-manager-dict-only-updated-one-level-deep
    #     manager = mp.Manager()
    #     proxy_results = manager.dict()
    #     for a in algs_to_run:
    #         proxy_results[a] = manager.list()
    #     algs_to_run = None

    #     # Create a lock, must be done before the pool
    #     l = mp.Lock()

    #     # create partial function for fixed inputs
    #     # partFunc = partial(self.run_individual_algs, self.graph, proxy_results)

    #     # limit number of processes and tasks, initalize mutex lock
    #     if self.num_processes is None:
    #         self.num_processes = math.floor(mp.cpu_count)/2    
    #     pool = mp.Pool(initializer=self.init, initargs=(l,), processes=self.num_processes, maxtasksperchild=50)
        
    #     # iterate over 
    #     t0 = timer()
    #     # Add try except finally clause
    #     try:
    #         for ad in pool.imap(self.run_individual_algs, enumerate(tasks), chunksize=int(number_of_jobs // (self.num_processes**2) + 1)):     
    #             proxy_results[ad[0]].append(ad[1])      
    #             bar_assign_job.next()
    #         bar_assign_job.finish()
    #         t1 = timer() - t0
    #     except Exception as e_:
    #         pool.terminate()
    #         raise e_
    #     finally:
    #         # good practice
    #         pool.close()
    #         pool.join()

    #     # convert proxy dict data into standard python
    #     final_results = {}
    #     for k,v in proxy_results.items():
    #         final_results[k] = list(v)
        
    #     # Save it!
    #     if self.save_to_file == True:
    #         # Create folder if it doesn't exist already
    #         if not os.path.exists(self.save_directory):
    #             os.makedirs(self.save_directory)
    #         with open(os.path.join(self.save_directory, self.results_filename), 'wb') as f:            
    #             pickle.dump(final_results, f)
    #     # return it!
    #     return final_results
        

