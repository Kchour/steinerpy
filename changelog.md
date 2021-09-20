# Changelog
All notable changes to this project will be documented in this file. Furthermore, missing details will be added as I recall them.

## [Unreleased] - HEAD
### Fixed
- ~~Attempting to fix reconstructed path and solution table not matching error~~This can only occur when using inadmissible heuristic
- Make sure upon merging, the merged component will loop over their feasible paths to see if any can be inserted in the the shortest path queue, prior to next nomination. This is required because merging causes a jump in the `gmin` value for a particular component. 

## [Unreleased] - 9d0a0d
### Fixed
- Fixed an issue where unit tests were not tearing down properly, so global config settings were being tainted from previous unit tests


## [Unreleased] - e1a379
### Added
- User may select their graph domain in `config.py`. Either "generic" for unstructured graphs or "grid" for grid-based 2D graphs
- Randomized terminal unittest for MAPF instances. However SteinLib still not passing.
- Data structures in `generic_search` to keep track of root and children nodes.
- New classes in the `common` module: `CustomHeuristic`, `GridBasedHeuristic2D`, and `HeuristicMap`. 
- Return gcosts, parent, children, and root dict in `context`
- setUp and tearDown to a couple unit tests that rely on heuristics

### Changed
- Allow S*-unmerged to detect paths in between components, instead of at the destination. Results in faster convergence
- The `shortest_path_check()` in unmerged now considers max(fmin1, fmin2, gmin1+gmin2)
- S*-unmerged functions nominate(), update(), and path_check() are no longer overriding; Directly use from `Framework`. 
- Animation in `Framework.py` to allow generic plotting.
- AnimateV2 class set axis equal syntax changed.
- In S*-unmerged, do not break out of solution handler if cycle detected; instead only break when tree criteria not satisfied.
- Progress Bar can now handle 1 instance correctly.
- Default grid-based heuristic is "diagonal_nonuniform" for mapf instances.
- unittest "test_steiner_sstar" is more complete now.
- To define a custom heuristic for example, we must now do:

    ```
    from steinerpy.algorithms.common import CustomHeuristics
    CustomHeuristics.bind(lambda next,goal: 1) 

    ```

### Removed
- Load of old comments.
- Removed hardcoded eps values from each S* shortest path criteria and in tree_criteria. The eps represents the least cost edge in the input graph. But we may not know this value in certain situations.

## [1.0.0] - Prior 2021-08-2
### Added
- TBD
### Changed
- TBD.
### Removed
- TBD.