#!/usr/bin/env python3

from graphs.graph import MyGraph
import matplotlib.pyplot as plt
import pdb


# # Create UNDIRECTED edge dict with weights (THIS IS EASIER)

edgeDict = {('v1','v2'): 1,
            ('v2','v3'): 1,
            ('v3','v4'): 1,
            ('v4','v5'): 1,
            ('v5','v6'): 1,
            ('v6','v7'): 1,
            ('v7','v8'): 1,
            ('v8','v5'): 1}

graph = MyGraph(edgeDict, "undirected")

pdb.set_trace()




