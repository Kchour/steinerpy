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
from .base import AFileHandle

class Process(AFileHandle):
    """Process generated results and compare with baseline
    
    Attributes:
       
    """

    def __init__(self, save_path: str="", file_behavior: str="HALT"):

        # creates self.file_behavior and self.save_path
        super().__init__(save_path=save_path, file_behavior=file_behavior)  

        # or specify either or both input objects
        self.baseline_data = None
        self.main_results_data = None

    def specify_files(self, baseline_file="", main_results_file=""):
        """User may specify the absolute path of either files
        """
        if baseline_file != "":
            with open(baseline_file, 'rb') as f:
                self.baseline_data = pickle.load(f) 

        if main_results_file != "":
            with open(main_results_file, 'rb') as f:
                self.main_results_data = pickle.load(f) 

    def specify_data(self, baseline_data=None, main_results_data=None):
        """User may simply pass in the data 
        """
        if baseline_data is not None:
            self.baseline_data = baseline_data

        if main_results_data is not None:
            self.main_results_data = main_results_data

    def _generate(self):
        """Generate a spreadsheet that summarizes all results

        sheet 1 contains final tree values
        sheet 2 contains statistics

        """
        # convert data to data frame (IGNORE THE TERMINALS FOR NOW)
        if self.baseline_data is not None:
            pd_baseline = pd.DataFrame(self.baseline_data["solution"]) 
            base_dist = [sum(pd_baseline['dist'][i]) for i in range(len(pd_baseline))]
            '''Create data frames'''
            # processs the base into df
            out_df = pd.DataFrame({
                'Baseline-Kruskal': base_dist
            })
        else:
            out_df = pd.DataFrame()

        # extract data comparing steiner tree values
        if self.main_results_data is not None:
            pd_res = pd.DataFrame(self.main_results_data["solution"])
            res_dist =  [   [sum(pd_res.iloc[i][name]['dist']) for name in pd_res.iloc[i].keys()] for i in range(len(pd_res))]

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
                    if j != "Kruskal":
                        statsDict[j] = pd_res.iloc[i][j]['stats']
                    elif self.baseline_data is not None and 'stats' in pd_baseline.iloc[i]:
                        statsDict[j] = pd_baseline.iloc[i]['stats']

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
        if self.save_path != "":
            dfList = [out_df, statsDF_Full]
            dfNames = ['results', 'stats']
            with pd.ExcelWriter(self.save_path) as writer:
                for df, name in zip(dfList, dfNames):
                    df.to_excel(writer, name)
                writer.save()

            # # Save basic results to excel
            # out_df.to_excel(os.path.join(directory, 'processed.xlsx'), "results")
            
            # # save stats to excel
            # statsDF_Full.to_excel(os.path.join(directory, 'processed.xlsx'), "stats")

            print("Process: wrote {}!".format(self.save_path))
            if cfg.Misc.sound_alert == True:
                os.system('spd-say "Finished! Wrote to spreadsheet!"')
        
        # return steiner tree values, and algorithm stats
        return out_df, statsDF_Full
