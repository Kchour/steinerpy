# TODO List
- Finish documenting modules and provide more examples of how to use library
- Consider using PRM* radius for proximity-based algorithm
- Need to get rid of warnings from animation libraries depending on matplotlib
- Add better support for directed edges
- Better termination handling when nomination queue is empty (we just throw an error atm)
- Dont call reconstruct path since we are keeping track of root nodes, unless we are trying to visualize! Currently, visualization isn't even supported for generic graphs.
- ~~Add support generic graphs~~
- ~~Remove unnecessary functions from `Common`~~
- ~~Maintain lexicographic ordering for tables to save space~~ Don't think we'll do this for now
- ~~Maintain global bounds (kruskal) via a priority queue heap~~
- Don't add all the randomized tests spreadsheets to git
- Consider using `lmin` to do a last-past scan of recently merged component's feasible paths.
- ~~Move 2D heuristics to a separate module...~~
- ~~Recosntructing path shouldn't duplicate any nodes~~
- ~~Reuse nearest neighbor heuristic...~~
- Store routines in Common for cleanliness
- Clean up logging
- ~~Reconstructing can be done without using search_utils~~
- ~~Rename GenericSearch to MultiSearch and also Create UniSearch. Move all of these to search_algorithms instead of generic_algorithms~~
- Need to use Generic Graph class for keeping track of feasible paths which can handle relabeling, deletion, and insertion of edges, nodes quite easily.
- Don't commit .xlsx files from randomized_tests
- Cleanup cfg files, too many classes there
- Need to separate tests into two use cases: normal end users and developers. 



