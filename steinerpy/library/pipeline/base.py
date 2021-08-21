from typing import Union, List, Tuple
import os
import logging
import random as rd

from steinerpy.library.graphs import IGraph

my_logger = logging.getLogger(__name__)

class Generate:
    def __init__(self, graph: IGraph, save_path: str="", file_behavior: Union["SKIP", "HALT", "RENAME", "OVERWRITE"]="HALT"):
        self.graph = graph
        self.save_path = save_path
        self.file_behavior = file_behavior
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

        minX, maxX, minY, maxY = self.graph.grid_dim

        list_of_instances = []
        for _ in range(num_of_inst):
            terminal_set = set()
            while len(terminal_set) < num_of_terms:
                # randomly generate a point
                pt = (rd.randint(minX, maxX), rd.randint(minY, maxY))
                # make sure point is unique using set and not an obstacle!
                if self.graph.obstacles is not None :
                    if  pt not in self.graph.obstacles:
                        terminal_set.add(pt)
                else:
                    terminal_set.add(pt)
            list_of_instances.append(list(terminal_set))

        return list_of_instances

    def run(self):
        """Behavior specified by `file_behavior`
        
        """
        if self.file_behavior == "HALT":
            # halt program if file exists already!
            if os.path.exists(self.save_path):
                raise FileExistsError('{} already exists!'.format(self.save_path))   
        elif self.file_behavior == "SKIP":
            # Simply move on, no need to halt
            if os.path.exists(self.save_path):
                my_logger.info("".join(self.save_path, "already exists, but skipping"))
        elif self.file_behavior == "OVERWRITE":
            # Ignore existing baseline file and just run
            return self._generate()
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
            return self._generate()  

    def _generate(self):
        """
            must return 
               {
                'terminals': self.instances,
                'solution': self.solution,
               }
        """
        pass