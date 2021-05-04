#!/usr/bin/env python3
# from search.search_utils import reconstruct_path
# from graphs.graph import SquareGrid
# from graphs.ogm import OccupancyGridMap
# import numpy as np

# import matplotlib.pyplot as plt

from search.search_utils import DoublyLinkedList
import pdb

dbl = DoublyLinkedList()
dbl.push(1)
dbl.push(2)
dbl.push(3)
dbl.append(4)
dbl.insert(10, 3, 2)

print("wip")
dbl.print_list(1, method="forward")
dbl.print_list(1, method="reverse")

pdb.set_trace()

