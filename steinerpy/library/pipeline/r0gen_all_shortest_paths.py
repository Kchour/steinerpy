# MARK FOR DELETION

import pickle
from timeit import default_timer as timer
import multiprocessing as mp
import numpy as np
import math
import os
from functools import partial

from steinerpy.library.search.search_algorithms import UniSearch
from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.library.misc.utils import Progress

class OfflinePaths:      
    """Class related to obtaining all shortest paths  """
    # Default method without any mp. VERY SLOW. DONT USE ONLY FOR SHOWCASE
    @classmethod
    def get_all_paths(cls, graph):
        data = {}
        if 'SquareGrid' in str(type(graph)):
            minX, maxX, minY, maxY = graph.grid_dim
            # loop over start
            for i in range(minX, maxX, graph.grid_size):
                for j in range(minY, maxY, graph.grid_size):
                    if graph.obstacles is not None and (i,j) in graph.obstacles:
                        continue
                    else:
                        search = UniSearch(graph, (i,j), None, 'zero', False) #(works)    
                        parents, g = search.use_algorithm()
                        data.update({
                                    (i,j):{
                                        'parents': parents,
                                        'g': g
                                    } 
                                })
        # elif 'MyGraph' in str(type(graph)):
        #     search = UniSearch(graph, i, j, 'zero', False)

        # Save data somewhere
        return data


    # Reformed function with speed up!
    @classmethod
    def get_all_paths_fast(cls, graph, num_processes=None, save_file=None):
        
        if save_file is not None and os.path.exists(save_file):
            raise FileExistsError('{} already exists!'.format(save_file))

        if num_processes is None:
            if mp.cpu_count() > 8:
                num_processes = mp.cpu_count
            else:
                num_processes = math.floor(mp.cpu_count()/2)

        data_pool = []
        pool = mp.Pool(processes=num_processes, maxtasksperchild=100)
        # Check for square grids only
        if 'SquareGrid' in str(type(graph)):
            # Get all non-occupied cells
            locs = np.where(graph.grid==0)
            # Create bar
            bar_assign_job = Progress(len(locs[0]))
            
            # create generator expression, perhaps save memory? remeber x,y are flipped because of how arrays indexing works
            locs_gen = ((x,y) for x,y in zip(locs[1], locs[0]))
            del locs

            # Create a proxy dict with manager
            manager = mp.Manager()
            final_results_pool = manager.dict()

            # create partial function because graph doesnt change
            partFunc = partial(OfflinePaths.get_all_paths_ind, graph, final_results_pool)
            # final_results_pool = {}

            # partFunc(855, 354)
            for res in pool.imap_unordered(partFunc,locs_gen, chunksize=1):
                # final_results_pool.update(res)
                bar_assign_job.next()

            # for x,y in zip(locs[0], locs[1]):
            #     bar_assign_job.next()
            #     # MAKE SURE start node is not an obstacle!
            #     # if graph.obstacles is not None and (x,y) in graph.obstacles:
            #     #     continue
            #     # else:
            #     # data_pool[(x,y)] = pool.apply(OfflinePaths.get_all_paths_fast, args=(sq, x, y))
            #     data_pool.append(pool.apply_async(OfflinePaths.get_all_paths_ind, args=(graph, x, y)))
            #     # Increment bar, doesn't work with nested loops...

            # # tell bar to finish up and close
            bar_assign_job.finish()

            # will this help?
            pool.close()
            pool.join()

        # # Now call get method. Added a progress bar
        # bar_job_get = IncrementalBar('Job completion progress', max = len(data_pool))
        # final_results_pool = {}
        # for ndx, p in enumerate(data_pool):
        #     # print("getting ndx {} out of {}".format(ndx, len(data_pool)))
        #     bar_job_get.next()
        #     # timer.sleep(1)
        #     final_results_pool.update(p.get())
        
        #    # bar finish!
        # bar_job_get.finish()

        # Save results to a specific directory if wanted
        if save_file is not None:
            with open(save_file, 'wb') as f:
                pickle.dump(final_results_pool, f)

        return final_results_pool

    # Allow mp by passing in a start location
    @classmethod
    def get_all_paths_ind(cls, graph, proxy_dict, start_node):
        startx, starty = start_node
        data = {}       
        search = UniSearch(graph, (startx,starty), None, 'zero', False) #(works)  
        _, g = search.use_algorithm()
        del search, _
        # This is too large?
        # data.update({
        #             (startx,starty):{
        #                 'g': g
        #             } 
        #         })
        #TEST 
        data.update({
                (startx,starty): ""
            })
        proxy_dict[start_node] = data

    ############################################################################

if __name__=="__main__":
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

    ### Now apply mp pooling
    # get all non-obstacle locations
    startTime = timer()
    OfflinePaths.get_all_paths_fast(sq)
    #end timer
    endTime = timer()
    print(endTime - startTime)
    print("")
    # print(final_results_pool)
    
    # Calculate all shortest paths
    startTime = timer()
    data = OfflinePaths.get_all_paths(graph=sq)
    endTime = timer()
    print(endTime - startTime)
    print("")

   