"""This script outputs a spreadsheet or prints a single instance to terminal. It is used to process our 'results' files, generated from a baseline

Workflow:
    1) First create a baseline file using 'r1generate_baseline.py'
        a) Also (optionally), creates obstacles for a baseline file using 'r1agenerate_baseline_obstacles.py'
    2) Generate results using a baseline file using 'r2generate_results.py'
    3) Run this script to create a spreadsheet based on the 'results' and 'baseline' files

Todo:
    - Encapsulate this script into a class    
    - Come up with better file names?

"""

import pickle
import os
import pandas as pd
import numpy as np
import re

import steinerpy.config as cfg

class Process:
    """Process generated results and compare with baseline
    
    Attributes:
        baseline_filename (str): Filename for the baseline file generated using kruskals
        results_filename (str): Filename for the results file generated using our algorithms
        output_filename (str): Filename for the processed file. Can be user specified, but defaults to parameters 
            based on baseline file
        baseline_dir (str): Absolute path to baseline files
        results_dir (str): Absolute path to result files
        out_dir (str): Absolute path for output files, produced by this class
        save_to_file (bool): Save file to disk if True
        base_data (dict): Baseline data produced using Kruskals
        results (dict): Generated results on baseline graphs

    """

    def __init__(self, baseline_dir, results_dir, out_dir=None, baseline_filename=None, results_filename=None, output_filename=None, save_to_file=True):
        self.baseline_filename = baseline_filename
        self.results_filename = results_filename
        self.output_filename = output_filename
        self.baseline_dir = baseline_dir
        self.results_dir = results_dir
       
        # By default, the output directory is same as the results dir
        if out_dir is None:
            self.out_dir = self.results_dir
        else:
            self.out_dir = out_dir

        self.save_to_file = save_to_file
        
        # Predefined output xlsx output file based on baselne file
        if save_to_file and output_filename is None:
            match = re.match(r'(^baseline_)(\w+.+)(.pkl)', baseline_filename)
            self.output_filename = 'processed_'+match.group(2)+'.xlsx'

        # initialize (allow user to set these directly)
        self.base_data = None
        self.results = None

        # Load from file if specified
        if self.baseline_filename is not None and self.results_filename is not None:
            # Load baseline, 
            # directory = os.path.dirname(os.path.realpath(__file__))
            # but change baseline file name according to your needs
            with open(os.path.join(self.baseline_dir, self.baseline_filename), 'rb') as f:
                self.base_data = pickle.load(f)
            '''base_data (dict): Contains data pertaining to the baseline file
                base_data={ 
                'solution': [{'sol':[], 'path':[], 'dist':[]}, {...}, ..., {}]
                'terminals':[ [(x,y),...,(xn,yn)],[...],[...],....,[] ]
                'obstacles': [(x,y),...,(xn,yn) ]     
                }
            '''

            # Load results
            with open(os.path.join(self.results_dir, self.results_filename), 'rb') as f:
                self.results = pickle.load(f)
            ''' results (dict): Contains data pertaining to the results
                results = {
                    'Astar':[{...runresults...}]
                        .
                        .
                        .
                    'Astar-Bidi':[{...runresults...}], 
                    'Primal-Dual': [{...runresults...}]
                }
                
                where,
                
                runresults={ 
                'solution': [{'sol':[], 'path':[], 'dist':[]}, {...}, ..., {}]
                }
            '''
            # Make sure results are non-empty
            for k,v in self.results.items():
                for i in range(len(self.results[k])):
                    assert(len(self.results[k][i]['sol'])>0)
                    assert(len(self.results[k][i]['dist'])>0)
                    assert(len(self.results[k][i]['path'])>0)
            

    def run_func(self):

        # lazy assign
        base_data = self.base_data
        results = self.results

        # FIXME Ignore kruskal obstacles for now. Need a better solution later
        if "obstacles" in base_data:
            del base_data['obstacles']

        # Convert both datasets to dataframe for ease of use
        pd_baseline = pd.DataFrame(base_data)
        pd_res = pd.DataFrame(results)

        # extract data comparing steiner tree values
        # use .iloc method to get an non-indexed row
        base_dist = [sum(pd_baseline['solution'][i]['dist']) for i in range(len(pd_baseline['solution']))]
        res_dist =  [   [sum(pd_res.iloc[i][name]['dist']) for name in pd_res.iloc[i].keys()] for i in range(len(pd_res))]

        '''Create data frames'''
        # processs the base into df
        out_df = pd.DataFrame({
            'Baseline-Kruskal': base_dist
        })

        # proccess the results into df (use assign with ** to add multiple columns)
        # To add a column to DF, unpack with ** operator on a dictionary
        np_res_dist = np.array(res_dist)
        out_df = out_df.assign(**{
                    name :np_res_dist[:,i]
                    for i, name in enumerate(pd_res.iloc[0].keys())
        })
        
        # Calculate %variation, add a column to df,   abs(1-x/avg)
        thresh = lambda x : 0.0 if (x < 0.0001) else x
        out_df = out_df.assign(**{
            'var': [thresh(abs(1.0 - sum(out_df.iloc[i])/(out_df.iloc[i][0])/len(out_df.iloc[0]))) for i in range(len(out_df))]
        })

        ##########################
        ### Check monotonicity ###
        ##########################
        # 0 if no problems, else 1!

        #INFO: The zip() function will only iterate over the smallest list passed.
        non_decreasing = lambda L: not all(x<=y+1e-3 for x,y in zip(L, L[1:]))

        mono = {}
        # iterate each row
        for i in range(len(pd_res)):
            # iterate over each label (Astar, Djikstra, etc..)
            for j in pd_res.iloc[i].keys():
                if j+'ND' not in mono:
                    # Initialize mono[j] if j not in mono
                    mono[j+'ND'] = [non_decreasing(pd_res.iloc[i][j]['dist'])]
                else:
                    # Just append if j exists
                    mono[j+'ND'].append(non_decreasing(pd_res.iloc[i][j]['dist']))
        # Add mono column to out_df
        out_df = out_df.assign(**mono)

        #################################
        ### Add STATS to a new DF     ###
        #################################

        # compute average
        avg = 0
        # data frame containing all stats
        statsDF_Full = pd.DataFrame()

        # create a list of all keys
        key_list = list(pd_res.iloc[0].keys())
        key_list.append("Kruskal")

        # iterate each row
        for i in range(len(pd_res)):
            # iterate over each label (Astar, Dijkstra, etc...)
            statsDict = {}
            for j in key_list:
                # get stats from our main algorithms (not including Kruskal)
                if j not in statsDict:
                    if j is not "Kruskal":
                        statsDict[j] = pd_res.iloc[i][j]['stats']
                    elif 'stats' in pd_baseline.iloc[i]['solution']:
                        statsDict[j] = pd_baseline.iloc[i]['solution']['stats']

            # convert to dataframe and transpose
            statsDF = pd.DataFrame(statsDict).T

            # Add iteration instance i as another index
            index_vals = list(statsDF.index)
            statsDF.index = pd.MultiIndex.from_product([['run {}'.format(i)], index_vals,])
            column_vals = list(statsDF.columns)

            # Add names as columns, reorder
            # statsDF['index'] = statsDF.index
            # df_vals = list(statsDF.columns.values)
            # statsDF = statsDF[df_vals[::-1]]

            # Add multindex, heirarchal title for iteration number
            # statsDF.columns = pd.MultiIndex.from_product([['iteration {}'.format(i)], df_vals[::-1] ])
            # statsDF.rows= pd.MultiIndex.from_product([['iteration {}'.format(i)], df_vals])

            # append to full thing
            # statsDF_Full = pd.concat([statsDF_Full, statsDF], ignore_index=False)
            statsDF_Full = statsDF_Full.append(statsDF, ignore_index = False) 
            # statsDF_Full = statsDF_Full.append(pd.Series(), ignore_index = True)

            # Get sum
            avg += np.array(statsDF)
            print("test {}".format(i))

        # Get average
        avg /= len(pd_res)
        finalIndex = pd.MultiIndex.from_product([['Average'], index_vals,])
        avgDF = pd.DataFrame(avg, index = finalIndex, columns=column_vals)
        statsDF_Full = statsDF_Full.append(avgDF, ignore_index = False)

        ########################################################################
        ### Write to excel, requires ExcelWriter for multiple DFs and sheets ###
        ########################################################################
        if self.save_to_file:
            dfList = [out_df, statsDF_Full]
            dfNames = ['results', 'stats']
            with pd.ExcelWriter(os.path.join(self.out_dir, self.output_filename)) as writer:
                for df, name in zip(dfList, dfNames):
                    df.to_excel(writer, name)
                writer.save()

            # # Save basic results to excel
            # out_df.to_excel(os.path.join(directory, 'processed.xlsx'), "results")
            
            # # save stats to excel
            # statsDF_Full.to_excel(os.path.join(directory, 'processed.xlsx'), "stats")

            print("wrote to spreadsheet!")
            if cfg.Misc.sound_alert == True:
                os.system('spd-say "Finished! Wrote to spreadsheet!"')
        
        # return steiner tree values, and algorithm stats
        return out_df, statsDF_Full

if __name__=="__main__":
    # Specify baseline file
    # baseline_filename = 'baseline_maze-128-128-2.map-2t-5i.pkl'
    baseline_filename = 'baseline_2t-10i.pkl'

    # split filename, get the description
    sp = baseline_filename.split('_')[1]
    results_filename = "results_{}".format(sp)

    sp2 = sp.split('.pkl')[0]
    output_filename = "processed_{}{}".format(sp2, '.xlsx')

    # specify directory to read baseline file and results file, and to write xlsx to
    # directory = os.path.dirname(os.path.realpath(__file__))+"/../"
    import steinerpy.config as cfg
    baseline_dir = cfg.results_dir + "/tests"
    results_dir = cfg.results_dir + "/tests"

    pr = Process(baseline_dir, results_dir, baseline_filename=baseline_filename, results_filename=results_filename, output_filename=output_filename)
    pr.run_func()