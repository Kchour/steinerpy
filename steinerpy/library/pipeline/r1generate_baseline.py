"""This module is used to produce baseline files """

import pickle
import random as rd
import os
from timeit import default_timer as timer
import multiprocessing as mp
import logging
from typing import  Union


from .base import Generate

# from steinerpy.library.logger import MyLogger
from steinerpy.context import Context
import steinerpy.config as cfg
from steinerpy.library.misc.utils import Progress
from steinerpy.library.graphs import IGraph

my_logger = logging.getLogger(__name__)

class GenerateBaseLine(Generate):
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

    def __init__(self, graph: IGraph, save_path: str="", file_behavior: Union["SKIP", "HALT", "RENAME", "OVERWRITE"]="HALT", 
                cache_dir: str="", load_from_disk=False):
        """
        This class will behave differently depending on the number and type of inputs

        Params:
            graph: object that must of IGraph type
            num_of_terms: number of terminals per instance
            num_of_inst: number of instances overall. 
            save_path: absolute location of output file containing baseline results
            file_behavior: only relevant if `save_filename` is specified
            cache_dir: absolute location of cache files, pertinent to current graph and output filename. 

        Todo:
            Figure out caching

        """        
        super().__init__(graph, save_path, file_behavior, load_from_disk)
        self.cache_dir = cache_dir   
        self.solution = []

    def _cache_lookup(self):
        """Use cache to avoid recomputing distances if specified
            TODO: Kruskal context cannot handle this yet

        """
        if self.cache_dir != "":
            # Read each cache file and modify the instances if necessary
            # Append number to cache if it exists
            cnt = 0
            temp = self.cache_dir
            self.cachefiles = []
            while True:
                if os.path.exists(temp + str(cnt).zfill(3)):
                    self.cachefiles.append(temp+str(cnt).zfill(3))
                    cnt += 1
                else:
                    break
            self.cache_count = cnt

    def _generate(self):
        """ Queries Kruskal's algorithm for each run instance for baseline results

        Results are saved to file and also returned!
        
        """       
        #start time
        t0 = timer()

        # Run Kruskals on each and save results
        my_logger.info("Running Kruskal's algorithm")  

        # keep track of progress
        pg = Progress(len(self.instances))
        for ndx, t in enumerate(self.instances):
            try:
                my_logger.info("Running (single-core) terminals: {}".format(t))

                # OPTIONALLY PASS CACHE FILE TO KRUSKAL CONTEXT
                context = Context(self.graph, t)
                context.run('Kruskal')
                self.solution.append(context.return_solutions())
            except:
                my_logger.warning("one or more terminals may be landlocked! In ndx {}, with terminals: {}".format(ndx,t), exc_info=True)

            pg.next()

        # finish progress bar, end line
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

            print("Finished! Wrote baseline file! Now generate results! {} secs".format(t1))
            if cfg.Misc.sound_alert == True:
                os.system('spd-say "Finished! Wrote baseline file! Now generate results!"')

        return {
                'terminals': self.instances,
                'solution': self.solution,
               }
