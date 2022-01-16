""" If you change the config variable before importing, then
    you can set it. However after importing, it cannot be changed
"""

import unittest
import numpy as np

from steinerpy.library.animation.animationV2 import AnimateV2

# from steinerpy import context

# set seed if desired
random.seed(12)

# Create square grid using GraphFactory
minX = -25			# [m]
maxX = 25   
minY = -25
maxY = 25
grid = None         # pre-existing 2d numpy array?
grid_size = 1       # grid fineness[m]
grid_dim = [minX, maxX, minY, maxY]
n_type = 8           # neighbor type

# Create a squareGrid using GraphFactory
sq = GraphFactory.create_graph("SquareGrid", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type)      
sq.name = "square"

# Load other maps: MAPF

mapf_maze = DataParser.parse(os.path.join(cfg.data_dir,"mapf", "maze-32-32-2.map"), dataset_type="mapf")
mapf_maze.name = "maze"
mapf_den = DataParser.parse(os.path.join(cfg.data_dir,"mapf", "den312d.map"), dataset_type="mapf")
mapf_den.name = "den"
# store maps
# test_maps_mapf = [sq, mapf_maze, mapf_den]
test_maps_mapf = [sq, mapf_maze, mapf_den]

# load steinlib graphs
test_maps_steinlib = []
test_terminals_steinlib = []
stein_dir = {}
for root, dirs, files in os.walk(os.path.join(cfg.data_dir, "steinlib", "B")):
    for fname in files:
        # ensure proper file extension
        if "stp" in fname:
            sl_g, sl_terminals = DataParser.parse(os.path.join(root, fname), dataset_type="steinlib")
            sl_g.name = fname

            # store 
            test_maps_steinlib.append(sl_g)
            test_terminals_steinlib.append(sl_terminals)
            stein_dir[fname] = {'dir': os.path.join(root, fname), 'map': sl_g, 'terminals': sl_terminals}

# location to save cache files! WARNING: generating cache can take a long time
temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")

# get current working directory (the test folder)
cwd = os.path.dirname(os.path.abspath(__file__))

class TestGenerateAndCompareResultsMAPFGridBase(unittest.TestCase):
    """Randomly generate terminals and see if the answers will match!
        You can have the option of skipping baseline generation
        if the "XXX_baseline.pkl" file has already been generated
        for a particular instance-term combination
    """

    def setUp(self):
        self.old_setting = cfg.Algorithm.sstar_heuristic_type
        cfg.Algorithm.sstar_heuristic_type = "diagonal_nonuniform"
        # cfg.Misc.log_conf["handlers"]['console']['level'] = "DEBUG"
        # cfg.reload_log_conf()
        # cfg.Animation.visualize = True

    def tearDown(self):
        cfg.Algorithm.sstar_heuristic_type = self.old_setting  

    @unittest.skip("not testing right now")
    def test_generate_randomized_terminals_results_compare_mapf(self):
        # This heuristic is good for 8-neighbor square grids
        cfg.Algorithm.sstar_heuristic_type = "diagonal_nonuniform"

        # try reprioritzing
        # cfg.Algorithm.reprioritize_after_sp = False       #default
        # cfg.Algorithm.reprioritize_after_merge = True       #default

        num_of_inst = 1
        num_of_terms = 50
        for _map in test_maps_mapf:
            baseline_save_path = os.path.join(cwd, "".join((_map.name, "_baseline.pkl")))
            gen_bs = GenerateBaseLine(graph=_map, save_path=baseline_save_path, file_behavior="OVERWRITE", load_from_disk=True)
            # generate random instances
            gen_bs.randomly_generate_instances(num_of_inst, num_of_terms)
            # run the generator
            kruskal_results = gen_bs.run()
            # save instances
            instances = gen_bs.instances    

            # Generate results
            main_save_path = os.path.join(cwd, "".join((_map.name, "_main_results.pkl")))
            algs_to_run = ["S*-BS", "S*-HS", "S*-MM", "S*-MM0", "S*-BS-UN", "S*-HS-UN", "S*-MM-UN", "S*-MM0-UN"]


            gen_rm = GenerateResultsMulti(graph=_map, save_path=main_save_path, file_behavior="OVERWRITE", algs_to_run=algs_to_run)
            # specify instances
            gen_rm.input_specifed_instances(instances)
            # run the generator
            main_results = gen_rm.run()

            # loop over instances and compare mst values
            for ndx in range(len(instances)):
                # store mst values for instance
                kruskal_value = sum(kruskal_results['solution'][ndx]['dist'])

                # loop over algorithms
                for alg in algs_to_run:
                    alg_value = sum(main_results['solution'][alg][ndx]['dist'])
                    # now compare all values in mst_values
                    try:
                        if abs(kruskal_value-alg_value) > 1e-6:
                            print (alg, main_results['terminals'][ndx], kruskal_value, alg_value)
                            raise ValueError("MST VALUES DONT MATCH")
                    except:
                        my_logger.error("much badness during 'test_generate_randomized_terminals_results_compare_mapf'", exc_info=True)
                        raise 

                # test for monotonicity
                try:
                    assert all( y-x>=0 for x,y in zip(main_results['solution'][alg][0]['dist'],main_results['solution'][alg][0]['dist'][1:] ))
                except:
                    print("alg {} dist {}".format(alg, main_results['solution'][alg][0]['dist']))
                    print("kruskal order: {}".format(kruskal_results['solution'][0]['dist'] ))

            # process them
            save_path = os.path.join(cwd, "".join((_map.name, "processed_rand_results_test.xlsx")))
            pr = Process(save_path, file_behavior="OVERWRITE")
            pr.specify_files(baseline_save_path, main_save_path)
            pr.run()


class TestGenerateRandomResultsSteinLibGenericGraph(unittest.TestCase):

    def setUp(self):
        self.old_setting = cfg.Algorithm.sstar_heuristic_type
        from steinerpy.heuristics import Heuristics
        cfg.Algorithm.graph_domain = "generic"
        
        self.old_setting_domain = cfg.Algorithm.graph_domain 

        cfg.Misc.log_conf["handlers"]['console']['level'] = "WARN"
        cfg.reload_log_conf()
        # cfg.Animation.visualize = True
        Heuristics.bind(lambda next, goal: 0)

    def tearDown(self):
        cfg.Algorithm.sstar_heuristic_type = self.old_setting  
        cfg.Algorithm.graph_domain = self.old_setting_domain

    @unittest.skip("some issues yet")
    def test_generate_randomized_terminals_results_compare_steinlib(self):
        # from steinerpy.algorithms.common import CustomHeuristics
        # cfg.Algorithm.graph_domain = "generic"
        # zero_h = lambda *x, **kwargs: 0
        # CustomHeuristics.bind(zero_h)

        for _map, _terminals in zip(test_maps_steinlib, test_terminals_steinlib):
            baseline_save_path = os.path.join(cwd, "".join((_map.name, "_baseline.pkl")))
            gen_bs = GenerateBaseLine(graph=_map, save_path=baseline_save_path, file_behavior="SKIP", load_from_disk=True)
            # Get terminals
            gen_bs.input_specifed_instances([_terminals])
            # run the generator
            kruskal_results = gen_bs.run()
            # save instances
            instances = gen_bs.instances    

            # Generate results
            main_save_path = os.path.join(cwd, "".join((_map.name, "_main_results.pkl")))
            algs_to_run = ["S*-BS", "S*-HS", "S*-MM", "S*-MM0", "S*-BS-UN", "S*-HS-UN", "S*-MM-UN", "S*-MM0-UN"]
            # algs_to_run = ["S*-BS-UN", "S*-MM-UN", "S*-MM0-UN", "S*-HS-UN"]
            # algs_to_run = ["S*-HS"]
            gen_rm = GenerateResultsMulti(graph=_map, save_path=main_save_path, file_behavior="OVERWRITE", algs_to_run=algs_to_run)
            # specify instances
            gen_rm.input_specifed_instances([_terminals])
            # run the generator
            main_results = gen_rm.run()

class TestConfig(unittest.TestCase):
    
    def test_update_and_cleandraw_xy(self):
        print("test_update_and_cleandraw_xy")
        xo = np.linspace(-15,15, 150)
        yo = np.cos(xo)
        zo = np.cos(yo)

        for x,y,z in zip(xo, yo, zo): 
            # Add things
            AnimateV2.add_line("cos", x, y, markersize=5, marker='o', zorder=10) #on top
            AnimateV2.add_line("sin", x, z, draw_clean=True, markersize=10, marker='o')
            
            # Update figure
            AnimateV2.update()

        # Close figure when done
        AnimateV2.close()

    def test_delete_xy(self):
        print("test_delete_xy")
        xo = np.linspace(-15,15, 150)
        yo = np.cos(xo)
        zo = np.sin(xo)
        to = np.sin(xo) + np.cos(xo)

        for a,b in enumerate(zip(xo,yo,zo,to)): 
            x,y,z,t = b
            # Add things
            AnimateV2.add_line("cos", x, y, markersize=5, marker='o', zorder=10) #on top
            AnimateV2.add_line("sin+cos", x, t, draw_clean=True, markersize=10, marker='o')
            
            # if a > half delete sin!
            if a >= 125:
                AnimateV2.delete("sin")
            else:
                AnimateV2.add_line("sin", x, z, draw_clean=True, markersize=10, marker='o')

            # Update figure
            AnimateV2.update()
        
        # Close figure when done
        AnimateV2.close()

    def test_variable_number_of_inputs(self):
        print("test_variable_number_of_inputs")
        xo = np.linspace(-15,15,150)
        yo = np.cos(xo)
        data = np.array([xo,yo]).T.tolist()

        for d in data:
            # Add artists
            AnimateV2.add_line('cos', d, markersize=5, marker='o')

            # Update canvas drawings      
            AnimateV2.update()

        AnimateV2.close()


    def test_multiple_xy_input(self):
        print("test_multiple_xy_input")
        xo = np.linspace(-15,15,150)
        yo = np.cos(xo)
        data = np.array([xo,yo]).T

        skip = 5
        for ndx in range(0, len(data), skip):
            if ndx + skip >= len(data):
                break
            # Add artists
            AnimateV2.add_line('cos', data[ndx:ndx+skip,0].tolist(), data[ndx:ndx+skip,1].tolist(), markersize=5, marker='o')

            # Update canvas drawings      
            AnimateV2.update()

        AnimateV2.close()

    def test_multiple_xy_input_with_args(self):
        print("test_multiple_xy_input")
        xo = np.linspace(-15,15,150)
        yo = np.cos(xo)
        data = np.array([xo,yo]).T

        skip = 5
        for ndx in range(0, len(data), skip):
            if ndx + skip >= len(data):
                break
            # Add artists
            AnimateV2.add_line('cos', data[ndx:ndx+skip,0].tolist(), data[ndx:ndx+skip,1].tolist(), 'ro', markersize=5)

            # Update canvas drawings      
            AnimateV2.update()

        AnimateV2.close()

    def test_marker_color_and_size(self):
        print("test_marker_color_and_size")
        xo = np.linspace(-15,15, 150)
        yo = np.cos(xo)
        zo = np.cos(yo)

        for x,y,z in zip(xo, yo, zo): 
            # Add things
            AnimateV2.add_line("cos", x, y, markersize=15, marker='o', color='b', zorder=10) #on top
            AnimateV2.add_line("sin", x, z, draw_clean=True, markersize=10, marker='o', color='r')
            
            # Update figure
            AnimateV2.update()

        # Close figure when done
        AnimateV2.close()

    def test_marker_color_and_size_with_args(self):
        print("test_marker_color_and_size_with_args")
        xo = np.linspace(-15,15, 150)
        yo = np.cos(xo)
        zo = np.cos(yo)

        for x,y,z in zip(xo, yo, zo): 
            # Add things
            AnimateV2.add_line("cos", x, y, 'bo', markersize=15, zorder=10) #on top
            AnimateV2.add_line("sin", x, z, 'o', draw_clean=True, markersize=10)
            
            # Update figure
            AnimateV2.update()

        # Close figure when done
        AnimateV2.close()

    def test_marker_only_plot_no_lines(self):
        print("test_marker_only_plot_no_lines")
        xo = np.linspace(-15,15, 5)
        yo = np.cos(xo)
        zo = np.cos(yo)

        for x,y,z in zip(xo, yo, zo): 
            # Add things
            AnimateV2.add_line("cos", x, y, 'bo', markersize=15, zorder=10) #on top
            AnimateV2.add_line("sin", x, z, 'o', draw_clean=True, markersize=10)
            
            # Update figure
            AnimateV2.update()

        # Close figure when done
        AnimateV2.close()

    def test_compact_data_with_args(self):
        print("test_compact_data_with_args")
        xo = np.linspace(-15,15,250)
        yo = np.cos(xo)
        data = np.array([xo,yo])

        # Add artists
        AnimateV2.add_line('cos', data.tolist(), 'ro', markersize=5)

        # Update canvas drawings      
        AnimateV2.update()

        # Close figure when done
        AnimateV2.close()

if __name__ == "__main__":
    unittest.main()