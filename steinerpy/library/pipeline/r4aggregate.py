""" This module is used to aggregate average data over all algorithm results """

import os
import pandas as pd
import re
import xlsxwriter
# import cloudpickle as cp

import steinerpy.config as cfg

class Aggregate:
    """Class to aggregate average results, generate figure inside spreadsheet

    load_directory (str): location of processed results to load
    save_directory (str): location for storing aggregated result

    """
    def __init__(self, load_directory, save_directory, output_filename="Aggregated.xlsx"):
        self.load_directory = load_directory
        self.save_directory = save_directory
        self.output_filename = output_filename

        # WARN ABOUT OVERWRITING FILE
        if os.path.exists(os.path.join(save_directory, self.output_filename)):
            raise FileExistsError('{} already exists!'.format(self.output_filename))

    def run_func(self):
        # Get a list of all files
        lfiles = os.listdir(self.load_directory)

        # Initialize container of dataframes
        data = {'Time': pd.DataFrame(), 'Expanded_nodes': pd.DataFrame()}

        # load the data from each file, but skip non-processed related files
        for f in lfiles:
            # only process relevant files
            if "processed_" in f:
                # load data
                with open(os.path.join(self.load_directory, f), 'rb') as lf:
                    # read excel, some of the data will be treated as series
                    temp_data = pd.read_excel(lf, sheet_name="stats", index_col=[0,1])
                    
                    # get map name only
                    # match = re.match(r'(^processed_)(.+)(.map.+)', f)
                    # map_name = match.group(2)
                    match = re.match(r'(^processed_)(.+)(.xlsx)', f)
                    map_name = match.group(2)

                    # Create a dataframe from map data
                    ta = pd.DataFrame({map_name: temp_data['time']['Average']}).T
                    te = pd.DataFrame({map_name: temp_data['expanded_nodes']['Average']}).T
                    
                    # store relevent data. Rmb pandas append returns an object
                    data['Time'] = data['Time'].append(ta) 
                    data['Expanded_nodes'] = data['Expanded_nodes'].append(te)  

        # Add Multilevel column indices
        data['Time'].columns = pd.MultiIndex.from_product([['Time'], data['Time'].columns])
        data['Expanded_nodes'].columns = pd.MultiIndex.from_product([['Expanded_nodes'], data['Expanded_nodes'].columns])

        # Write to xlsx file!
        startrow=0
        with pd.ExcelWriter(os.path.join(self.save_directory, self.output_filename), engine='xlsxwriter') as writer:

            # to excel routine
            for df in data.values():
                df.to_excel(writer, startrow=startrow)

                # prepare for adding charts
                workbook = writer.book
                worksheet = writer.sheets['Sheet1']
                # follow requires xlsxwriter
                chart = workbook.add_chart({'type': 'column'})

                # Loop through algorithm (col)
                # name:= series' name (which shows up in legend)
                # values:= series' values 
                # categories:= Labels for each group of series
                for col_num in range(1, df.shape[1]+1):
                    chart.add_series({
                        'name': ['Sheet1', startrow+1, col_num],
                        'values': ['Sheet1',startrow+3, col_num, startrow+3+df.shape[0], col_num],
                        'gap': 500,
                        'categories': ['Sheet1', startrow+3, 0, startrow+3+df.shape[0], 0],
                    })
                # Set log scale?
                # chart.set_y_axis({'log_base': 10})
                chart.set_y_axis({
                    'minor_gridlines': {
                        'visible': True,
                        'line': {'width': 1.25, 'dash_type': 'square_dot'}
                    },
                })
                ###### chart.set_title(['Sheet1', startrow, 0]) ####
                # insert chart the specified location in xlsx
                worksheet.insert_chart('J{}'.format(startrow+1), chart, {'x_scale': 1.25, 'y_scale': 1.25})



                 # Try adding the reverse of the previous chart
                chart = workbook.add_chart({'type': 'column'})
                for row_num in range(startrow+3, startrow+3+df.shape[0]):
                    chart.add_series({
                        'name': ['Sheet1', row_num, 0],
                        'values': ['Sheet1', row_num, 1 ,row_num, 1+df.shape[1]],
                        'gap':500,
                        'categories': ['Sheet1',startrow+1, 1, startrow+1, 1+df.shape[1]]
                    })
                # chart.set_y_axis({'log_base': 10})
                chart.set_y_axis({
                    'minor_gridlines': {
                        'visible': True,
                        'line': {'width': 1.25, 'dash_type': 'square_dot'}
                    },
                })
                worksheet.insert_chart('T{}'.format(startrow+1), chart, {'x_scale': 1.25, 'y_scale': 1.25})


                # change start row indexing for the second object
                startrow += df.shape[0]
                startrow += 5

            # end writer
            writer.save()
        print("Finished Aggregating")
            
if __name__ == "__main__":
    load_dir = cfg.results_dir + "/mapf_run_1"
    save_dir = load_dir

    ag = Aggregate(load_dir, save_dir)
    ag.run_func()

