"""Generate results using S* algorithms """

import pickle
import os
from timeit import default_timer as timer
import re
import math
import itertools as it
import multiprocessing as mp
from progress.bar import IncrementalBar
from functools import partial

from steinerpy.library.logger import MyLogger
from steinerpy.context import Context
import steinerpy.config as cfg

class GenerateResults:
    """Generate results based on baseline files, so that we can process the results.

    Note: 
        Prior to running this file, the baseline file must already exist. If not,
        run 'r1generate_baseline.py' or 'r1agenerate_baseline_obstacles.py'.     

    Args:
        graph (SquareGrid, MyGraph): Graph classes from superclass IGraph.
            Created using 'GraphFactory' class from the 'graph' module   
        load_directory (str): Location to load baseline files
        baseline_file (str): Name of baseline file in the previous directory
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
    def __init__(self, graph, load_directory=None, baseline_file=None, save_directory=None, save_filename=None, run_instance=None, save_to_file=False, file_behavior="SKIP"):
        self.graph = graph
        self.minX, self.maxX, self.minY, self.maxY = graph.grid_dim
        self.baseline_file = baseline_file
        self.run_instance = run_instance
        self.save_to_file = save_to_file
        self.load_directory = load_directory

        # By default, the save directory is the same as load_directory
        if save_directory is None:
            self.save_directory = load_directory
        else:
            self.save_directory = save_directory

        if self.save_to_file:
            if save_filename is None:
                # We have to obey strict file-naming schemes for this to work
                match = re.match(r'(baseline_)(\w+.+)\.pkl', baseline_file)
                # sp = baseline_file.split('_')[1]
                # self.results_filename = "results_{}".format(sp)
                if match:
                    self.results_filename = "results_"+match.group(2)+".pkl"
                else:
                    raise AssertionError("no matching results filename for {}".format(baseline_file))
            else:
                self.results_filename = save_filename
            # Get path and check for existance. Don't overwrite files!
            # directory = os.path.dirname(os.path.realpath(__file__))
            
            self.SKIP = False
            if os.path.exists(os.path.join(self.save_directory, self.results_filename)):
                if not file_behavior=="SKIP":
                    raise FileExistsError('{} already exists!'.format(self.results_filename))
                
                self.SKIP = True 
 
        # Load Kruskal Baseline data/terminals
        # directory = os.path.dirname(os.path.realpath(__file__))
        self.data = {
            'terminals': [],
            'solutions': {
                'dist': [],
                'path': [],
                'sol': []
            },
            'obstacles': []
        }
        if self.load_directory is not None and self.baseline_file is not None:
            # terminals: list of "list of tuples"
            # solution: list of {'dist': [], 'path': [], 'sol': []}
            # 
            with open(os.path.join(self.load_directory, self.baseline_file), 'rb') as f:
                self.data = pickle.load(f)
                print("")

    def set_terminals(self, terminal_list):
        """terminal_list is a list of lists of tuples, i.e. [[(x1,y1), ... ,(xn, yn)], ..., [(a1,b1), ... , (am, bm)]]"""
        self.data["terminals"] = terminal_list

    def get_terminals(self):
        """Simple function to return terminals"""
        return self.data['terminals']

    def run_func(self, algorithms_to_run=None):
        """Run each of our algorithms using a context class

        The context class takes a 'graph' and an instance of 't' terminals, 
        then executes each algorithm on a given instance.
        Results are appended to the appropriate location in the 'results' attribute.
        Finally, output files have a  'results_' string prepended in the current directory
        for processing

        Returns:
            results (dict): Contains results for the algorithms ran

        """
        t0 = timer()
        # Use contextualizer to run algorithms (make sure debug visualizer is off for speed)
        if algorithms_to_run is None:
            results = {'SstarAstar':[], 'SstarPrimalDual': [], 'Astar':[],  \
                    'SstarMM':[]}
            algorithms_to_run = list(results)
        else:
            results = {alg: [] for alg in algorithms_to_run}
        
        # results = {'Astar':[], 'SstarAstar':[], 'SstarDijkstra': [], 'SstarPrimalDual': []}


        for ndx, t in enumerate(self.data['terminals']):

            if self.run_instance is not None and ndx != self.run_instance:
                # Skip everything except user-specified run
                continue

            # Add obstacles to graph?
            if 'obstacles' in self.data and self.data['obstacles'] and self.graph.obstacles is not None:
                # self.graph.set_obstacles(self.data['obstacles'][ndx])
                self.graph.set_obstacles(self.data['obstacles'])

            # Log terminals
            MyLogger.add_message("terminals: {}".format(t), __name__, "INFO")

            # Create context
            context = Context(self.graph, t)

            # context.run('Astar')
            # results['Astar'].append(context.return_solutions())

            for alg in algorithms_to_run:
                context.run(alg)
                results[alg].append(context.return_solutions())
            # context.run('SstarMM')
            # results['SstarMM'].append(context.return_solutions())
                      
            # context.run('SstarAstar')
            # results['SstarAstar'].append(context.return_solutions())

            # context.run('SstarPrimalDual')
            # results['SstarPrimalDual'].append(context.return_solutions())

            #We already have kruskals as base line
            print('finished with instance {} \n'.format(ndx))

        t1 = timer() - t0   

        # Save results!
        if self.save_to_file:
            # directory = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(self.save_directory, self.results_filename), 'wb') as f:
                pickle.dump(results, f)

            print("Wrote results to file! Now process them! {} secs".format(t1))
            if cfg.Misc.sound_alert is True:
                os.system('spd-say "Finished! Wrote results to file! Now process them!"')
        
        return results

class GenerateResultsMulti(GenerateResults):
    def __init__(self, graph, load_directory, baseline_file, save_directory=None, save_filename=None, run_instance=None, save_to_file=True, num_processes=4, file_behavior="OVERWRITE"):
        super().__init__(graph, load_directory, baseline_file, save_directory, save_filename, run_instance, save_to_file)
        self.num_processes = num_processes

    def init(self,l):
        """Required to avoid race conditions for lists

        Info:
            https://stackoverflow.com/questions/25557686/python-sharing-a-lock-between-processes

        """
        global lock
        lock = l

    def run_func(self): 
      if not self.SKIP: 
        # Create task generator, progress bar
        algs_to_run = ['SstarAstar', 'SstarPrimalDual', 'SstarMM', "Astar"]
        number_of_jobs = len(list(it.product(self.data['terminals'], algs_to_run)))
        bar_assign_job = IncrementalBar('Job progress', max = number_of_jobs)
        tasks = it.product(self.data['terminals'], algs_to_run)
        # unload data attribute for memory relief
        self.data = None

        # create Manager for shared object between processes
        # https://stackoverflow.com/questions/41780322/why-is-a-manager-dict-only-updated-one-level-deep
        manager = mp.Manager()
        proxy_results = manager.dict()
        for a in algs_to_run:
            proxy_results[a] = manager.list()
        algs_to_run = None

        # Create a lock, must be done before the pool
        l = mp.Lock()

        # create partial function for fixed inputs
        # partFunc = partial(self.run_individual_algs, self.graph, proxy_results)

        # limit number of processes and tasks, initalize mutex lock
        if self.num_processes is None:
            self.num_processes = math.floor(mp.cpu_count)/2    
        pool = mp.Pool(initializer=self.init, initargs=(l,), processes=self.num_processes, maxtasksperchild=50)
        
        # iterate over 
        t0 = timer()
        # Add try except finally clause
        try:
            for ad in pool.imap(self.run_individual_algs, enumerate(tasks), chunksize=int(number_of_jobs // (self.num_processes**2) + 1)):     
                proxy_results[ad[0]].append(ad[1])      
                bar_assign_job.next()
            bar_assign_job.finish()
            t1 = timer() - t0
        except Exception as e_:
            pool.terminate()
            raise e_
        finally:
            # good practice
            pool.close()
            pool.join()

        # convert proxy dict data into standard python
        final_results = {}
        for k,v in proxy_results.items():
            final_results[k] = list(v)
        
        # Save it!
        if self.save_to_file == True:
            # Create folder if it doesn't exist already
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)
            with open(os.path.join(self.save_directory, self.results_filename), 'wb') as f:            
                pickle.dump(final_results, f)
        # return it!
        return final_results
        

    def run_individual_algs(self, inputs):
        job_id, data = inputs
        # print("starting job id: ", job_id)
        terminals, alg = data
        context = Context(self.graph, terminals)
        context.run(alg)
        # print("finished job id: ", job_id)

        # Try mutex locking to avoid race conditions
        # lock.acquire()
        # proxy_dict[alg].append(context.return_solutions())
        # lock.release()
        return alg, context.return_solutions()