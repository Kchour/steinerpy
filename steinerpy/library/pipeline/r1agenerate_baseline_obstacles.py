'''Read baseline file and add obstacles to it'''

import os
import cloudpickle 
import random as rd
import time
import numpy as np

from steinerpy.algorithms.kruskal import Kruskal
from steinerpy.context import Context

class GenerateBaselineObstacles:
    def __init__(self, graph, p_obstacles, load_directory, baseline_file='baseline.pkl', out_filename=None):
        self.graph = graph
        self.minX, self.maxX, self.minY, self.maxY = graph.grid_dim
        self.load_directory = load_directory
        
        if p_obstacles == 0 or p_obstacles is None or not p_obstacles:
            raise ValueError("Number of obstacles to be added must be nonzero!") 
        self.Pobs = p_obstacles
        

        # Load Kruskal Baseline data
        ''' Select a file to use
        
        baseline_100t-5i.pkl:       5 instances of 100-terminals
        baseline_5t-100i.pkl:       100 instances of 5-terminals
        baseline_10t-200i.pkl:      200 instances of 10-terminals
        baseline_5t-1000i.pkl:      1000 instances of 5-terminals
        baseline_5t-1500i.pkl:      1500 instances of 5-terminals
        baseline_5t-10000i.pkl:     10000 instances of 5-terminals
        baseline_100t-100i.pkl:     100 instances of 100-terminals
        '''

        # directory = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(self.load_directory, baseline_file), 'rb') as f:
            self.data = cloudpickle.load(f)

        '''data={ 
            'solution': [{'sol':[], 'path':[], 'dist':[]}, {...}, ..., {}]
            'terminals':[ [(x,y),...,(xn,yn)],[...],[...],....,[] ]
            'obstacles': [(x,y),...,(xn,yn) ]       # Assumed not changing
            }
        '''

        # Get instance number from file
        self.instances = len(self.data['terminals'])
        self.N = len(self.data['terminals'][0])
        if out_filename is None and self.Pobs != 0:
            self.out_filename = 'baseline_{}t-{}i-{}o.pkl'.format(self.N, self.instances, p_obstacles)
        else:
            self.out_filename = out_filename

    # Generate line of obstacles 
    def while_func(self, terminals):
        array = set()   

        y = round(self.maxY/2)
        line = [(x,y) for x in np.arange(round(0.5*self.minX),round(0.5*self.maxX),self.graph.grid_size)]

        y = round(-self.maxY/2)
        line.extend([(x,y) for x in np.arange(round(0.5*self.minX),round(0.5*self.maxX),self.graph.grid_size)])

        x = 0
        line.extend([(x,y) for y in np.arange(round(0.5*self.minY), round(0.5*self.maxY), self.graph.grid_size)])

        for pt in line:
            if pt not in terminals:
                array.add(pt)
            if len(array) == self.Pobs:
                break
        yield array

    ''' run kruskal with the obstacles '''
    def run_func(self):
        #Start time
        t0 = time.time()

        # Get path and check for existance. Don't overwrite files!
        # directory = os.path.dirname(os.path.realpath(__file__))
        if os.path.exists(os.path.join(self.load_directory, self.out_filename)):
            raise FileExistsError('{} already exists!'.format(self.out_filename))

        obstacles = [[list(i) for i in self.while_func(self.data['terminals'][j])][0] for j in range(self.instances)]
        terminals = self.data['terminals']

         # Run Kruskals on each and save results
        solution = []
        for ndx, (t, o) in enumerate(zip(terminals, obstacles)):
            self.graph.set_obstacles(o)
            ko = Kruskal(self.graph, t)
            ko.run_algorithm()
            solution.append(ko.return_solutions())
            print(ndx, '\n')

        # end time
        t1 = time.time() - t0   

        # dump instances and solution to file
        # directory = os.getcwd()
        with open(os.path.join(self.load_directory, self.out_filename), 'wb') as f:
            cloudpickle.dump({
                'terminals': terminals,
                'solution': solution,
                'obstacles': obstacles
            }, f)

        print("Finished! Wrote obstacles-baseline file! Now generate results! {}".format(t1))
        os.system('spd-say "Finished! Wrote baseline file! Now generate results!"')

if __name__ =="__main__":
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

    # generate baseline with obstacles based on baseline files
    N = 2
    instances = 10
    obstacles = 1

    # specify directory to read baseline file and to write obstacles to
    # directory = os.path.dirname(os.path.realpath(__file__))+"/../"
    from steinerpy.library.config import Config as cfg
    load_directory = cfg.results_dir

    #create and run. Input baseline file to read
    gb = GenerateBaselineObstacles(graph, obstacles, load_directory, 'baseline_{}t-{}i.pkl'.format(N,instances))
    gb.run_func()