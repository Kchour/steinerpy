'''
    Steps
    1) run 'generate_baseline.py' to create baseline.pkl
    2) run this file
'''

import pickle
import os
import unittest

from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.context import Context

class TestAlgorithmsBaseLine(unittest.TestCase):
    def test_alg_baseline(self):
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
        try:
            graph = GraphFactory.create_graph("SquareGrid", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type)      
            self.assertTrue(True)
        except Exception as _e:
            raise _e

        # Load Kruskal Baseline data
        directory = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(directory, 'baseline_test_single.pkl'), 'rb') as f:
            data = pickle.load(f)

        '''data={ 
            'solution': [{'sol':[], 'path':[], 'dist':[]}, {...}, ..., {}]
            'terminals':[ [(x,y),...,(xn,yn)],[...],[...],....,[] ]
            }
        '''

        # Use contextualizer to run algorithms (make sure debug visualizer is off)
        results = {
            'S*-HS-UN': [],
            'S*-HS':[],  
        }

        # Fill 'Kruskal' results from baseline file
        for ndx, t in enumerate(data['terminals']):
            # Create context
            context = Context(graph, t)

            # astar unmerged
            context.run('S*-HS-UN')
            results['S*-HS-UN'].append(context.return_solutions())

            # formerly bi-directional astar
            context.run('S*-HS')
            results['S*-HS'].append(context.return_solutions())
            #We already have kruskals as base line

        # Save results!
        directory = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(directory, 'results_test_single.pkl'), 'wb') as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    unittest.main()