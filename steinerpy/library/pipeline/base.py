from typing import Union, List, Tuple
import os
import logging
import random as rd
import pickle
import numpy as np

from steinerpy.library.graphs import IGraph
from steinerpy.env_type import EnvType

my_logger = logging.getLogger(__name__)




class AFileHandle:
    """ A virtual class for file handling
        Load from disk only applicable during "skip" file behavior
    """
    def __init__(self, save_path:str, file_behavior:str, load_from_disk=False):
        self.file_behavior = file_behavior
        self.save_path = save_path
        self.results = None
        self.load_from_disk = load_from_disk

    def run(self):
        """Behavior specified by `file_behavior`
        
        """
        if self.file_behavior == "HALT":
            # halt program if file exists already!
            if os.path.exists(self.save_path):
                raise FileExistsError('{} already exists!'.format(self.save_path))   
            else:
                self.results = self._generate()
                return self.results
        elif self.file_behavior == "SKIP":
            if os.path.exists(self.save_path):
                # Simply move on, (try to load previously generated results if desired)
                my_logger.info("".join((self.save_path, "already exists, skipping")))
                # try loading from disk if enabled
                if self.load_from_disk == True:
                    self.results = self._load()
                    # # make sure instances are loaded from disk too
                    # self.instances = self.results['terminals']
                    return self.results
            else:
                # if path is unique, then generate results
                self.results = self._generate()
                return self.results
        elif self.file_behavior == "OVERWRITE":
            # Ignore existing baseline file and just run
            self.results = self._generate()
            return self.results
        elif self.file_behavior == "RENAME":
            # rename output file by appending a counter. Increment by 1 if previous counter value exists!
            cnt = 1
            while True:
                temp = self.save_path
                if os.path.exists(temp):
                    temp += str(cnt)
                    cnt += 1
                else:
                    self.save_path = temp
                    break
            self.results = self._generate()
            return self.results

    def _load(self):
        """Load result from disk if desired

        Make sure to return some results obj
        
        """
        # with open(self.save_path, 'rb') as f:
        #     my_logger.info("Loading results from file {}".format(self.save_path))
        #     results = pickle.load(f)
        
        # return results
        pass
    def _generate(self):
        """Run your generator here i.e. create results, output file etc.
        """
        pass
    
class Generate(AFileHandle):
    """Class for generating problem instances, results"""
    def __init__(self, graph: IGraph, save_path: str="", file_behavior: Union["SKIP", "HALT", "RENAME", "OVERWRITE"]="HALT", load_from_disk=False):
        super().__init__(save_path, file_behavior, load_from_disk)
        
        self.graph = graph
        self.instances: List[List[Tuple]] = None

    def randomly_generate_instances(self, num_of_inst: int, num_of_terms: int):
        """Randomly generate 2d points within the grid-based graph, ensuring it is not an obstacle

        Todo:
            Extend sampling to generic graphs!

        """
        self.instances = self._generate_random_instances_func(num_of_inst, num_of_terms)

    def input_specifed_instances(self, instances:List[List[Tuple]]):
        """User can specify their own problem instances, pertinent to the input graph

        """
        self.instances = instances

    def _generate_random_instances_func(self, num_of_inst: int, num_of_terms: int):

        list_of_instances = []
        for _ in range(num_of_inst):
            list_of_instances.append(self.graph.sample_uniform(num_of_terms))
            # TODO: need to compute up with sample_uniform method for generic graphs

        return list_of_instances

    def _load(self):
        """Load result from disk if desired
        
        """
        with open(self.save_path, 'rb') as f:
            my_logger.info("Loading results from file {}".format(self.save_path))
            results = pickle.load(f)
            # make sure instances are loaded from disk too
            self.instances = results['terminals']
        
        return results

    def _generate(self):
        """
            must return the following 
               {
                'terminals': self.instances,
                'solution': self.solution,
               }
        """
        pass