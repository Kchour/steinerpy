"""This module is used to parse different online datasets and convert them to a format we can use """
import re
from steinerpy.library.graphs.graph import GraphFactory
import pdb

class DataParser:
    """This class provides an interface for selecting a dataset to parse

    """
    @classmethod
    def parse(cls, filename: str, dataset_type: str = "steinlib", depots: list = None):
        """Parse the input file line-by-line

        Args:
            filename (str): Location of data file
            dataset_type (str): "steinlib", "mapf"

        Returns:
            graph (MyGraph), terminalList (list of list): If dataset_type is "steinlib"
            graph (SquareGrid): If dataset_type is  "mapf" 

        Todo:
            * Need to define more complicated features like water or swamp for mapf

        """
        # Read file
        with open(filename, 'r') as f:
            # read file contents and generate a list of each line 
            lines = f.readlines()	

        if dataset_type == "steinlib":
            # Check for edges (raw)
            edgePattern = r'(^E)\s(\d+)\s(\d+)\s(\d+)'
            # check for terminals
            terminalPattern = r'(^T)\s(\d+)'
            # patterns collection
            patterns = {'edge': edgePattern, 'terminal': terminalPattern}
            
            # return a dictionary
            edgeDict, terminalList = cls._regex_steinlib(lines, patterns)

            # Use GraphFactory class to create desired graph           
            graph = GraphFactory.create_graph(type_="Generic", edge_dict=edgeDict, graph_type="undirected")

            # return graph and terminalList
            print("parsed steinlib file!")

            return graph, terminalList

        elif dataset_type == "mapf" or dataset_type=="grid_2d":
            # define some regex patterns
            heightPattern = r'(^height)\s(\d+)'
            widthPattern = r'(^width)\s(\d+)'
            mapPattern = r'\.+|@+|T+|G+'
            
            # define other non-regex patterns
            obs = ['@', 'T']
            trav = ['.', 'G']

            # patterns
            patterns = {
                'regex': {
                    'height': heightPattern, 
                    'width': widthPattern,
                    'map': mapPattern
                },
                'non-regex':{
                    'obstacles': obs, 
                    'traverse': trav
                }
            }

            # Do the actual line-by-line parsing
            obs_coords, height, width = cls._regex_mapf(lines, patterns)

            # required SquareGrid definitions
            minX, maxX = 0, width - 1
            minY, maxY = 0, height - 1

            # assumed grid fineness of 1
            grid_size = 1      
            grid_dim = [minX, maxX, minY, maxY]
            
            # neighbor type (4 or 8)
            n_type = 8    
            
            # use GraphFactory class to create desired graph
            if depots is None:
                graph = GraphFactory.create_graph("SquareGrid", grid_dim=grid_dim, grid_size=grid_size, n_type=n_type, obstacles=obs_coords)
                # g.show_grid() #debugging
            else:
                graph = GraphFactory.create_graph("SquareGridDepot", grid_dim=grid_dim, grid_size=grid_size, n_type=n_type, obstacles=obs_coords, depots=depots)


            return graph

        elif dataset_type == "grid_3d":
            # 3d graphs may get really large, so let's rely on a database            

            pass
            max_x, max_y, max_z = [int(v) for v in lines[0].strip().split(" ") if v.isdigit()]
            grid_dim = [0, max_x, 0, max_y, 0, max_z]
            obstacles = []
            for line in lines[1::]:
                ox, oy, oz = [int(v) for v in line.strip().split(" ")]
                obstacles.append((ox, oy, oz))

            graph = GraphFactory.create_graph("SquareGrid3D", grid_dim, grid_size=1, obstacles=obstacles)
            return graph
            



    @staticmethod
    def _regex_mapf(lines, patterns):
        """A method for parsing MAPF instances from movingai.com

        Args:
            lines (list of str): A list of line-by-line text from a file
            patterns (dict): Contains 'regex' and 'non-regex' patterns for processing lines

        Returns:
            obs_coords (list of list), height (int), width (int): 

        """
        grid = []
        height = None
        width = None

        for line in lines:
            # cleanup the line
            line = line.strip()

            for name, p in patterns['regex'].items():
                match = re.search(p, line)

                # Change behavior depending on match
                if match:
                    if name == 'height':
                        height = int(match.group(2))
                    elif name == 'width':
                        width = int(match.group(2))                
                    elif name == 'map':
                        grid.append(list(line))

        # process with numpy
        import numpy as np
        test = np.array(grid)

        # get indices of obstacles
        obsMask = (test == patterns['non-regex']['obstacles'][0]) + (test == patterns['non-regex']['obstacles'][1])
        
        # flatten
        ind = np.where(obsMask==True)

        # flip x/y to show origin at bottom left. convert to list       
        obs_coords = np.vstack((ind[1], ind[0])).T.tolist()

        # convert to list of tuples
        obs_coords = [tuple(i) for i in obs_coords]

        # return 
        return obs_coords, height, width

    @staticmethod 
    def _regex_steinlib(lines, patterns):
        """Method used to parse steinlib instances

        Args:
            lines (list of str): A list of line-by-line text from a file
            patterns (dict): Contains 'edge' and 'terminal' regex patterns

        Returns:
            edgeDict (dict), terminalList (list): if steinlib

        """
        # dictionary for storing edges
        edgeDict = {}
        terminalList = []

        # iterate over each line
        for line in lines:
            # cleanup the line (remove trailing and leading spaces)
            line = line.strip()
            # test each pattern, but continue on the first match!
            for name, p in patterns.items():
                match = re.search(p, line)

                if match:
                    print(line)
                    if name == 'edge':
                        edgeDict[(match.group(2), match.group(3))] = float(match.group(4))
                    elif name == 'terminal':
                        terminalList.append( (match.group(2)) )
        
        return edgeDict, terminalList

