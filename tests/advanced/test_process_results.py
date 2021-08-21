from steinerpy.library.pipeline.r3process import Process
from steinerpy.library.misc.utils import Progress
import unittest
import os

from steinerpy.library.graphs.graph import GraphFactory
import steinerpy.config as cfg

# Create square grid using GraphFactory
minX = -15			# [m]
maxX = 15   
minY = -15
maxY = 15
grid = None         # pre-existing 2d numpy array?
grid_size = 1       # grid fineness[m]
grid_dim = [minX, maxX, minY, maxY]
n_type = 8           # neighbor type

# Create a squareGrid using GraphFactory
sq = GraphFactory.create_graph("SquareGrid", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type)      

class TestProcessResults(unittest.TestCase):
    
    # If pandas is installed, perform this unittest
    try:
        from steinerpy.library.pipeline.r3process import Process
        # def test_process_results_multicore(self):
        #     """Runs 20 instances of 50 terminals """
        #     baseline_directory = os.path.dirname(__file__)
        #     results_directory = os.path.dirname(__file__)
        #     baseline_filename = 'baseline_test_multi.pkl'
        #     results_filename = 'results_test_multi.pkl'
        #     pr = Process(baseline_directory, results_directory, baseline_filename=baseline_filename,
        #         results_filename=results_filename)
        #     pr.run_func()

        # def test_process_results_sequentially(self):
        #     """Runs 5 instances of 5 terminals """
        #     baseline_directory = os.path.dirname(__file__)
        #     results_directory = os.path.dirname(__file__)
        #     baseline_filename = 'baseline_test_single.pkl'
        #     results_filename = 'results_test_single.pkl'
        #     pr = Process(baseline_directory, results_directory, baseline_filename=baseline_filename,
        #         results_filename=results_filename)
        #     pr.run_func()

        def test_process_some_data(self):
            # get current working directory (the test folder)
            cwd = os.path.dirname(os.path.abspath(__file__))


            # THEY MUST ALSO BE THE SAME LENGTH!
            baseline_file = os.path.join(cwd, "baseline_test_single.pkl")
            main_results_file = os.path.join(cwd, "main_results_test_multi.pkl")

            # output format must be .xlsx
            save_path = os.path.join(cwd, "processed_all_results_test.xlsx")
            pr = Process(save_path, file_behavior="OVERWRITE")

            # load files
            pr.specify_files(baseline_file, main_results_file)

            # now perform the processing
            pr.run()

    except ImportError as e:
        pass  # module doesn't exist, deal with it.

if __name__ == "__main__":
    unittest.main()

