import os
import unittest
import pickle

class TestReadPickle(unittest.TestCase):
    def test_baseline_size(self):
        directory = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(directory, 'baseline_test_single.pkl'), 'rb') as f:
            data = pickle.load(f)

        '''data={ 
            'solution': [{'sol':[], 'path':[], 'dist':[]}, {...}, ..., {}]
            'terminals':[ [(x,y),...,(xn,yn)],[...],[...],....,[] ]
            }
        '''
        # make sure data is not empty
        self.assertTrue(len(data['terminals'][0])>0)
        self.assertTrue(len(data['solution'][0]['sol'])>0)
        # self.assertTrue(len(data['solution'][0]['path'])>0)
        self.assertTrue(len(data['solution'][0]['dist'])>0)
   
    def test_results_size(self):
        # load baseline data for kruskal        
        directory = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(directory, 'baseline_test_single.pkl'), 'rb') as f:
            data = pickle.load(f)

        # baseline dist array
        baseline_dist = data['solution']

        directory = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(directory, 'results_test_single.pkl'), 'rb') as f:
            data = pickle.load(f)

        '''
            data = {
                'S*-unmerged':[results, {...}, ...,  {...}], 
                'S*-HS':[results, {...}, ...,  {...}], 
                'S*-HS0': [results, {...}, ...,  {...} ],
           }
            
            where
            
            results={'sol':[], 'path':[], 'dist':[]}
            }
        '''
        # make sure data is not empty anywhere
        for k,v in data.items():
            for i in range(len(data[k])):
                self.assertTrue(len(data[k][i]['sol'])>0)
                self.assertTrue(len(data[k][i]['dist'])>0)
                self.assertTrue(len(data[k][i]['path'])>0)

        #make sure distance results are the same between all algorithms (to within a tolerance)
        print("wip")
        eps = 1e-6
        # iterate over number of run-instances
        for i in range(len(data['S*-unmerged'])):
            alike_test = set()
            # iterate over algorithm names
            for k in data.keys():
                alike_test.add(sum(data[k][i]['dist']))
        mean = sum(alike_test)/len(alike_test)
        for val in alike_test:
            self.assertTrue( abs(val-mean)<0.001)
        self.assertTrue(abs(list(alike_test)[0] - sum(baseline_dist[i]['dist']) < eps))

if __name__ == "__main__":
    unittest.main()