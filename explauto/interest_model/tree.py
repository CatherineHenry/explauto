import math
import os
from itertools import combinations
from math import cos, sin, radians

from typing import List

from sklearn.metrics.pairwise import cosine_similarity
import copy

import time
import numpy as np
import random

import matplotlib.pyplot as plt

from datetime import datetime
from heapq import heappop, heappush

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Polygon, Wedge
from scipy.spatial.kdtree import minkowski_distance_p

from ..utils.annotations3d import annotate3D
from ..utils.plot_object import PlotObject
from ..utils.utils import rand_bounds
from ..utils.config import make_configuration
from .interest_model import InterestModel
from .competences import competence_exp, prediction_error_cos_dist_exp, competence_cos_dist_exp
from ..utils.observer import Observable
from ..utils.plot_object import PlotObject

from cozmo.nav_memory_map import NodeContentTypes

class InterestTree(InterestModel, Observable):
    """
    class InterestTree implements either R-IAC or SAGG-RIAC
    """
    def __init__(self, 
                 conf, 
                 expl_dims, 
                 max_points_per_region, 
                 max_depth,
                 split_mode, 
                 competence_measure, 
                 progress_win_size, 
                 progress_measure, 
                 sampling_mode,
                 plot_objects=None,
                 rand_seed=None,
                 region_deletion=False):

        self.rand_seed = rand_seed
        self.region_deletion_rng = np.random.default_rng(rand_seed)

        self.conf = conf
        self.bounds = self.conf.bounds[:, expl_dims]
        self.competence_measure = competence_measure

        if progress_win_size >= max_points_per_region:
            raise ValueError("WARNING: progress_win_size should be < max_points_per_region")

        self.data_x = None # list of target motor or sensory goals 'x'
        self.data_y = None # list of reached sensory effect
        self.data_c = None # list of competence measures
        self.data_nav_memory_map = None # list of navigation memory maps
        self.data_flow_uuid = None # list of flow ids
        self.region_deletion = region_deletion
        self.tree = Tree(self.get_data_x,
                         np.array(self.bounds, dtype=float),
                         self.get_data_y,
                         self.get_data_flow_uuid,
                         self.get_data_c,
                         self.get_data_nav_memory_map,
                         max_points_per_region=max_points_per_region,
                         max_depth=max_depth,
                         split_mode=split_mode,
                         progress_win_size=progress_win_size,
                         progress_measure=progress_measure,
                         sampling_mode=sampling_mode,
                         idxs=[],
                         plot_objects=plot_objects,
                         region_deletion_rng=self.region_deletion_rng)
        
        InterestModel.__init__(self, expl_dims)
        Observable.__init__(self)

    def get_data_x(self):
        return self.data_x

    def get_data_y(self):
        return self.data_y

    def get_data_c(self):
        return self.data_c

    def get_data_flow_uuid(self):
        return self.data_flow_uuid

    def get_data_nav_memory_map(self):
        return self.data_nav_memory_map
    
    def sample(self):
        # TODO: if it can't find a single node to sample that has free points then the program should end
        sampled_points = self.tree.sample()
        sample_attempts = 0
        while sampled_points is None:
            sample_attempts += 1
            print(f"Was not able to sample point, trying to sample again {sample_attempts}/{self.tree.n_children}")
            sampled_points = self.tree.sample()
            if sample_attempts > self.tree.n_children:
                print(f"Attempted to sample {sample_attempts} times. Exiting program")
                raise IndexError
        return sampled_points

    def progress(self):
        return self.tree.progress
    
    def max_leaf_progress(self):
        return self.tree.max_leaf_progress

    def random_walk_region_deletion(self, tree, pathing=None):
        # Will recurse until it either hits a leaf or a region is deleted
        if pathing is None:
            pathing = []
        if tree.leafnode: # we shouldn't get here except for root level
            # print("hit a leaf, skipped region deletion")
            return None
        # todo: idk if I like this, since it has a ton of early movement because with only 4 regions almost every point results in a region with no lower lower
        # elif tree.lower.lower is None or np.random.uniform() < 0.15: # automatically increase likelihood of nuking as we get deeper into tree bc Bernoulli Process
        # only enters random walk 30% of the time, as it walks there is only 20% chance of removing upper and lower nodes
        # elif np.random.uniform() < 0.2: # automatically increase likelihood of nuking as we get deeper into tree bc Bernoulli Process
        elif self.region_deletion_rng.random() < 0.2: # automatically increase likelihood of nuking as we get deeper into tree bc Bernoulli Process
            tree.leafnode = True
            tree.lower = None
            tree.greater = None
            print(f"[{datetime.now()}] region deleted! Split {tree.split_value} along dimension {tree.split_dim} with density {tree.density()} and pathing: {pathing}")
            return {'split_value': tree.split_value, 'split_dim': tree.split_dim, 'density': tree.density()}
        # elif np.random.uniform() < 0.5:
        #     pathing.append("lower")
        #     random_walk(tree.lower, pathing)
        # else:
        #     pathing.append("greater")
        #     random_walk(tree.greater, pathing)
        # Travel down path with the greatest density (this deviates from true random)
        if tree.lower.density() > tree.greater.density():
            pathing.append(f"lower (density: {tree.lower.density()})")
            return self.random_walk_region_deletion(tree.lower, pathing)
        else:
            pathing.append(f"greater (density: {tree.greater.density()})")
            return self.random_walk_region_deletion(tree.greater, pathing)

    def update(self, xy, ms, flow_uuid=None, nav_memory_map=None):
        """
        data_x will be either the motor vector or sensory vector depending on exploration dimensions.
        :param xy: Target SM Space (concat Motor x Sensory vectors)
        :param ms: Reached SM Space (concat Motor x Sensory vectors)
        :return:
        """
        # print np.shape(self.data_x), np.shape(np.array([xy[self.expl_dims]]))
        if self.data_x is None:
            self.data_x = np.array([xy[self.expl_dims]])
        else:
            self.data_x = np.append(self.data_x, np.array([xy[self.expl_dims]]), axis=0)
        if self.competence_measure is prediction_error_cos_dist_exp:
            cos_sim, cos_dist, competence = self.competence_measure(xy, ms)
            self.emit(f"[{flow_uuid}] competence", f"[cos sim: {cos_sim}, cos dist: {cos_dist}] bounded cos distance between target and reached: {competence}")
        elif self.competence_measure is competence_exp or self.competence_measure is competence_cos_dist_exp:
            competence = self.competence_measure(xy, ms)
            self.emit(f"[{flow_uuid}] competence", f"competence {competence}")
        if self.data_c is None:
            self.data_c = np.array([competence]) # Either prediction error or competence error
        else:
            self.data_c = np.append(self.data_c, competence)

        if self.data_y is None:
            self.data_y = np.array([ms[~np.isin(np.arange(len(ms)), self.expl_dims)]]) # sensory space is the non-expl_dims of the ms
        else:
            self.data_y = np.append(self.data_y, np.array([ms[~np.isin(np.arange(len(ms)), self.expl_dims)]]), axis=0)

        if self.data_flow_uuid is None: # keep track of flow uuids for training WAC classifier later (simplifies syncing the data for each subspace)
            self.data_flow_uuid = np.array([flow_uuid])
        else:
            self.data_flow_uuid = np.append(self.data_flow_uuid, np.array([flow_uuid]), axis=0)

        if self.data_nav_memory_map is None: # keep track of flow uuids for training WAC classifier later (simplifies syncing the data for each subspace)
            self.data_nav_memory_map = np.array([nav_memory_map])
        else:
            self.data_nav_memory_map = np.append(self.data_nav_memory_map, np.array([nav_memory_map]), axis=0)


        # Catherine TODO: is this the right spot for this?
        # TODO: put in so only triggers if experiment f, so have it default to never (0) unless f is set, in which case use f
        if self.region_deletion:
            # if np.random.uniform() < 0.3: # 30% of time traverse the tree and possibly delete a region
            if self.region_deletion_rng.random() < 0.3: # 30% of time traverse the tree and possibly delete a region
                explosions = self.random_walk_region_deletion(self.tree)

        self.tree.add(np.shape(self.data_x)[0] - 1)


## Essentially treating every node as a "tree". Leaf nodes identified by self.leafnode = True
class Tree(Observable):
    """
        Competence Progress Tree (recursive)
        
        This class provides an index into a set of k-dimensional points which
        can be used to rapidly look up the nearest neighbors of any point.
    
        Parameters
        ----------
        get_data_x : (N,K) array
            Function that return the data points to be indexed.
        bounds_x : (2,K) array
            Bounds on tree's domain ([mins,maxs])
        get_data_c : (N,K) array_like
            Function that return the data points' competences.
        max_points_per_region : int
            Maximum number of points per region. A given region is splited when this number is exceeded.
        max_depth : int
            Maximum depth of the tree
        split_mode : string
            Mode to split a region: 
                'random': random value between first and last points, 
                'median': median of the points in the region on the split dimension, 
                'middle': middle of the region on the split dimension, 
                'best_interest_diff': 
                    value that maximize the difference of progress in the 2 sub-regions
                    (described in Baranes2012: Active Learning of Inverse Models 
                    with Intrinsically Motivated Goal Exploration in Robots)
        progress_win_size : int
            Number of last points taken into account for progress computation (should be < max_points_per_region)
        progress_measure : string
            How to compute progress: 
                'abs_deriv_cov': approach from explauto's discrete progress interest model
                'abs_deriv': absolute difference between first and last points in the window, 
                'abs_deriv_smooth', absolute difference between first and last half of the window 
        sampling_mode : list 
            How to sample a point in the tree: 
                dict(multiscale=bool, 
                    volume=bool, 
                    mode=greedy'|'random'|'epsilon_greedy'|'softmax', 
                    param=float)                    
                multiscale: if we choose between all the nodes of the tree to sample a goal, leading to a multi-scale resolution
                            (described in Baranes2012: Active Learning of Inverse Models 
                            with Intrinsically Motivated Goal Exploration in Robots)
                volume: if we weight the progress of nodes with their volume to choose between them
                        (new approach)
                mode: sampling mode
                param: a parameter of the sampling mode: eps for eps_greedy, temperature for softmax.                                                 
        idxs : list 
            List of indices to start with
        split_dim : int
            Dimension on which the next split will take place
        
        Raises
        ------
        RuntimeError
            The maximum recursion limit can be exceeded for large data
            sets.  If this happens, either increase the value for the `max_points_per_region`
            parameter or increase the recursion limit by::
    
                >>> import sys
                >>> sys.setrecursionlimit(10000)
    

    """
    def __init__(self,
                 get_data_x,
                 bounds_x,
                 get_data_y,
                 get_data_flow_uuid,
                 get_data_c,
                 get_data_nav_memory_map,
                 max_points_per_region,
                 max_depth,
                 split_mode,
                 progress_win_size,
                 progress_measure,
                 sampling_mode,
                 idxs=None,
                 split_dim=0,
                 plot_objects=None,
                 region_deletion_rng=None
                 ):

        self.region_deletion_rng = region_deletion_rng
        self.get_data_x = get_data_x
        self.bounds_x = np.array(bounds_x, dtype=np.float64)
        self.get_data_y = get_data_y
        self.get_data_flow_uuid = get_data_flow_uuid
        self.get_data_c = get_data_c
        self.get_data_nav_memory_map = get_data_nav_memory_map
        self.max_points_per_region = max_points_per_region
        self.max_depth = max_depth
        self.split_mode = split_mode
        self.progress_win_size = progress_win_size
        self.progress_measure = progress_measure
        self.sampling_mode = sampling_mode

        self.plot_objects = [] if plot_objects is None else plot_objects
        self.split_dim = split_dim
        self.split_value = None
        self.lower = None
        self.greater = None
        if idxs == None:
            self.idxs = []
        else:
            self.idxs = idxs
        self.n_children = len(self.idxs)
        self.volume = np.prod(self.bounds_x[1,:] - self.bounds_x[0,:])
        
        self.leafnode = True # identifies if self is a Leaf Node
        self.can_sample = True # If there are no free spaces to travel to in the region this is set to false and the leaf is passed over when sampling
        self.progress = 0 # potential learning progress (will select points where this is high)
        self.max_leaf_progress = 0
        if self.n_children > self.max_points_per_region:
            self.split()
        self.update_max_progress()

        Observable.__init__(self)


    def get_nodes(self):
        """
        Get the list of all nodes.
        
        """
        # Inline functions here best aligns with original implementation using lambda, and allows for sharing the fold_up function, but there is probably a better way
        def add_lower_and_greater_with_parent(fl, fg):
            return [self] + fl + fg

        def leaf_as_list(leaf):
            return [leaf]

        return self.fold_up(f_inter=add_lower_and_greater_with_parent, f_leaf=leaf_as_list)


    def get_leaves(self, only_sampleable=False):
        """
        Get the list of all leaves.
        """

        # Inline functions here best aligns with original implementation using lambda, and allows for sharing the fold_up function, but there is probably a better way
        def add_lower_and_greater(fl, fg):
            return fl + fg

        def leaf_as_list(leaf):
                return [leaf]

        if only_sampleable:
            return self.fold_up_sampleable(f_inter=add_lower_and_greater, f_leaf=leaf_as_list)
        else:
            return self.fold_up(f_inter=add_lower_and_greater, f_leaf=leaf_as_list)

        # Note: Removed lambda so we can checkpoint the model (can't pickle tree with lambda iirc)
        # return self.fold_up(lambda n, fl, fg: fl + fg, lambda leaf: [leaf])


    # TODO: Can't save tree pickle w/ lambdas. fixed checkpointing by removing. Uncomment if we need (Note: was used in test_tree script).
    # def depth(self):
    #     """
    #     Compute the depth of the tree (depth of a leaf=0).
    #
    #     """
    #     return self.fold_up(f_inter=lambda n, fl, fg: max(fl + 1, fg + 1), f_leaf=lambda leaf: 0)
    #
    
    def density(self):
        """
        Compute the density of the node.
        
        """
        return self.n_children / self.volume
    
    
    def pt2leaf(self, x):
        """
        Get the leaf which domain contains x.
        
        """
        if self.leafnode:
            return self
        else:
            if x[self.split_dim] < self.split_value:
                return self.lower.pt2leaf(x)
            else:
                return self.greater.pt2leaf(x)
        
    def get_safe_coordinates_within_region_bounds(self):
        # Get the coordinates of all the leaf nodes that are safe (object/cliff/edge free) from the  Nav Mem Map
        safe_coordinate_regions = []
        unsafe_coordinate_regions = []
        latest_nav_map = self.get_data_nav_memory_map()[-1]

        latest_nav_map.quad_tree_safe_and_unsafe_coordinates(latest_nav_map.root_node, safe_coordinate_regions, unsafe_coordinate_regions)

        # pad unsafe coordinate regions so they have a border large enough to prevent selecting an action that would result in robot collision
        # It appears if a pose is selected where the robot would overlap/collide with the object, path planning is cancelled and shortest path is used
        for unsafe_coordinate_region in unsafe_coordinate_regions:
            # min x and y
            unsafe_coordinate_region[0] = unsafe_coordinate_region[0] - 60 # 40 is arbitrary ... testing and seeing whats works
            # max x and y
            unsafe_coordinate_region[1] = unsafe_coordinate_region[1] + 60

        safe_coordinate_regions_avoiding_collision = []
        dropped_safe_coordinates = []
         # trim safe coordinates so they don't overlap with the padded unsafe coordinates
        for safe_coordinate in safe_coordinate_regions:
            safe_region_min = safe_coordinate[0] # x,y coords of region min
            safe_region_max = safe_coordinate[1] # x,y coords of region max
            coordinate_is_safe_from_collision = True
            for unsafe_coordinate in unsafe_coordinate_regions:
                unsafe_region_min = unsafe_coordinate[0] # x,y coords of region min
                unsafe_region_max = unsafe_coordinate[1] # x,y coords of region max

                if (safe_region_min[0] >= unsafe_region_min[0] and safe_region_max[0] <= unsafe_region_max[0] and safe_region_min[1] >= unsafe_region_min[1] and safe_region_max[1] <= unsafe_region_max[1]):
                    # Safe navmap region is entirely inside of the padded unsafe region
                    # No trimming will help
                    # The region isn't free from collision with a point, no reason to keep checking other regions
                    coordinate_is_safe_from_collision = False
                    dropped_safe_coordinates.append(safe_coordinate)
                    break
                else:
                    # there is either no overlap or there is partial overlap. If partial, trim to fit
                    x_intersection_min = max(safe_region_min[0], unsafe_region_min[0])
                    y_intersection_min = max(safe_region_min[1], unsafe_region_min[1])
                    x_intersection_max = min(safe_region_max[0], unsafe_region_max[0])
                    y_intersection_max = min(safe_region_max[1], unsafe_region_max[1])

                    # check if the intersection results in invalid regions
                    if (x_intersection_min >= x_intersection_max or y_intersection_min >= y_intersection_max):
                        # The region is fully free from collision with a point, no reason to keep checking other regions
                        continue
                    else:

                        # We have the intersection, so for first pass, just the rectangle who's new corner is the intersection corner.
                        # TODO: Can explore dividing trimmed rectangle into three and adding to the list but then would need to check those new rectangles
                        # against all padded as well, and updating list while iterating is not ideal

                        # calculate size of intersection
                        x_intersection_size = abs(x_intersection_max - x_intersection_min)
                        y_intersection_size = abs(y_intersection_max - y_intersection_min)


                        # check if min and max x match intersect min/max x to see if we only shift up/down
                        if x_intersection_min == safe_region_min[0] and x_intersection_max == safe_region_max[0]:
                            # if y is shared on region_max and intersect_max, the region is lower than the intersect
                            if y_intersection_max == safe_region_max[1]: # lower than intersect
                                # only shift max y down
                                safe_region_max[1] = safe_region_max[1] - y_intersection_size
                            else: # above intersect
                                # only shift min y up
                                safe_region_min[1] = safe_region_min[1] + y_intersection_size

                        # check if min/max y match intersect min/max y to see if we only shift left/right
                        if y_intersection_min == safe_region_min[1] and y_intersection_max == safe_region_max[1]:
                            if x_intersection_min == safe_region_min[0]: # to the right of intersect
                                    # shift x min right
                                    safe_region_min[0] = safe_region_min[0] + x_intersection_size
                            else: # to left of intersect
                                # only shift x max left
                                safe_region_max[0] = safe_region_max[0] + x_intersection_size
                        # if x is shared on the region_min and intersect_min, the region is to the right of the intersect
                        if x_intersection_min == safe_region_min[0]: # to the right of intersect
                            # if y is shared on region_max and intersect_max, the region is lower than the intersect
                            if y_intersection_max == safe_region_max[1]: # right and lower than intersect
                                # min x shifts right by x intersection size
                                safe_region_min[0] = safe_region_min[0] + x_intersection_size
                                # shit max y down by y intersection size
                                safe_region_max[1] = safe_region_max[1] - y_intersection_size
                            else: # right and above intersect
                                # min x and y are shifted right by x and y intersection size
                                safe_region_min[0] = safe_region_min[0] + x_intersection_size
                                safe_region_min[1] = safe_region_min[1] + y_intersection_size
                                # max coordinates are unchanged

                        else: # is located to the left of the intersect
                            # if y is shared on region_max and intersect_max, the region is lower than the intersect
                            if y_intersection_max == safe_region_max[1]: # left and lower than intersect
                                # min coordinates are unchanged

                                # max x and y are shifted left and down by x and y intersection sizes
                                safe_region_max[0] = safe_region_max[0] - x_intersection_size
                                safe_region_max[1] = safe_region_max[1] - y_intersection_size

                            else: # left and above intersect
                                # min y shifts up by y intersection size
                                safe_region_min[1] = safe_region_min[1] + y_intersection_size
                                # max x shifts left by x intersection size
                                safe_region_max[0] = safe_region_max[0] - x_intersection_size

                        # update safe_coordinate so future checks take it into consideration. Any previously passed checks would have either
                        # been similarly trimmed or fully outside the collision boundary so not impacted by this change
                        safe_coordinate = np.array([[safe_region_min[0], safe_region_min[1]],[safe_region_max[0], safe_region_max[1]]])

                        # print(f"trimmed coordinate: {coordinate} to {intersection_coordinate}")
                        # safe_coordinate_regions_avoiding_collision.append(intersection_coordinate)
            if coordinate_is_safe_from_collision:
                safe_coordinate_regions_avoiding_collision.append(safe_coordinate)


        # Select all the safe leaf node coordinates from Nav Mem Map that are in the bounds of the region being sampled
        # Any intersecting coorinates are trimmed to fit within the region being sampled.
        min_bounds = self.bounds_x[0, :]
        max_bounds = self.bounds_x[1, :]
        safe_coordinate_regions_in_bounds = []
        for coordinate in safe_coordinate_regions_avoiding_collision:
            region_min = coordinate[0] # x,y coords of region min
            region_max = coordinate[1] # x,y coords of region max
            if (region_min[0] >= min_bounds[0] and region_max[0] <= max_bounds[0] and region_min[1] >= min_bounds[1] and region_max[1] <= max_bounds[1]):
                # Safe navmap region is entirely inside interest tree region to be sampled
                # No trimming needed
                safe_coordinate_regions_in_bounds.append(coordinate)
            else:
                # there is either no overlap or there is partial overlap. If partial, trim to fit
                x_intersection_min = max(region_min[0], min_bounds[0])
                y_intersection_min = max(region_min[1], min_bounds[1])
                x_intersection_max = min(region_max[0], max_bounds[0])
                y_intersection_max = min(region_max[1], max_bounds[1])

                # check if the intersection results in invalid regions
                if (x_intersection_min >= x_intersection_max or y_intersection_min >= y_intersection_max):
                    continue
                else:

                    intersection_coordinate = np.array([[x_intersection_min, y_intersection_min],[x_intersection_max, y_intersection_max]])
                    # print(f"trimmed coordinate: {coordinate} to {intersection_coordinate}")
                    safe_coordinate_regions_in_bounds.append(intersection_coordinate)
        return safe_coordinate_regions_in_bounds

    def sample_bounds(self):
        """
        Sample a point in the region of this node. If blocked points is not None then sample until
        a point is found that is not blocked.

        Note: Cannot build list of guaranteed unblocked to sample because floats, would be massive number of choices
        Note: May need to rework if passing becomes too large. should I pass taken or free? if I pass free I can do guarantee sample..
              or maybe I should just pass nav map and compute here

        1. nvm don't pass list, pass the nav map and check the type of the point sampled
            1.a.: what do we do if the entire node is object? We don't have a good way to back out of that situation
                1.b could back out and sample random other node instead?

        passing nav map (via robot world) by reference so it will be up to date when querying later


        TODO: this is going to take forever unless I reduce search to the known bounds, even if the grid is currently for all possible actions
        """
        if not self.can_sample:
            return None

        # Nav memory map stores x,y information for things in the space.
        # Cozmo pose is x,y _in front of cozmo_ (not center of cozmo) so they tell rotational information as well.
        # "The coordinate space is relative to Cozmo, where Cozmo's origin is the point on the ground between Cozmo's two front wheels"
        # Since the robot understands position by monitoring its tread movement,
        # it does not understand movement in the z axis. This means that the only
        # applicable elements of pose in this situation are position.x position.y
        # and rotation.angle_z.
        # x,y are in mm it looks like. So measure mm of exploration space for min/max

        safe_coordinate_regions_in_bounds = self.get_safe_coordinates_within_region_bounds()

        # if there are no safe coordinate regions in bounds, then we cannot sample this region
        if len(safe_coordinate_regions_in_bounds) == 0:
            self.can_sample = False
            min_bounds = self.bounds_x[0, :]
            max_bounds = self.bounds_x[1, :]
            print(f"No safe coordinates in bounds (dimension {self.split_dim} min: {min_bounds} max: {max_bounds})!")
            return None

        # pick a random safe Nav Mem Map leaf to sample a single coordinate from
        random_safe_leaf = random.choice(safe_coordinate_regions_in_bounds)
        random_safe_leaf_min_max_diff = random_safe_leaf[1, :] - random_safe_leaf[0, :]
        random_in_diff = random_safe_leaf_min_max_diff * np.random.rand(1, random_safe_leaf.shape[1])
        # add the min safe coordinate back to the random difference to get a final random safe coordinate within the bounds
        random_safe_coordinate =  random_in_diff + random_safe_leaf[0, :]

        # Sample a random point in region bounds to get a rotation (and any other dimensions of motor action that weren't restricted
        # by safe spaces in the NavMemMap) to add to the bounded motor action sampled from the Nav Memory Map
        rand_sample_in_region_bounds = rand_bounds(self.bounds_x).flatten()
        random_safe_coordinate_with_rotation = np.append(random_safe_coordinate, (rand_sample_in_region_bounds[2:]))
        return random_safe_coordinate_with_rotation

    def sample_bounds_exclude_objects(self):
        s = rand_bounds(self.bounds_x).flatten()

        return s
    
    def sample_random(self):
        """
        Sample a point in a random leaf.
        
        """
        if self.sampling_mode['volume']:
            # Choose a leaf weighted by volume, randomly
            if self.leafnode:
                return self.sample_bounds()
            else:
                if self.lower.can_sample is True and self.greater.can_sample is True:
                    split_ratio = ((self.split_value - self.bounds_x[0,self.split_dim]) /
                                   (self.bounds_x[1,self.split_dim] - self.bounds_x[0,self.split_dim]))

                    if split_ratio > np.random.random(): # TODO: does this really result 'weighted by volume' in practice?
                        return self.lower.sample(sampling_mode={'mode':'random'})
                    else:
                        return self.greater.sample(sampling_mode={'mode':'random'})
                elif self.lower.can_sample is False and self.greater.can_sample is False:
                    print("Cannot sample lower or greater child node, sample current instead")
                    return self.sample_bounds()
                elif self.lower.can_sample:
                    return self.lower.sample(sampling_mode={'mode':'random'})
                elif self.greater.can_sample:
                    return self.greater.sample(sampling_mode={'mode':'random'})

        else: 
            # Choose a leaf randomly
            return np.random.choice(self.get_leaves(only_sampleable=True)).sample_bounds()

        
    def sample_greedy(self):
        """        
        Sample a point in the leaf with the max progress.
        
        """    
        if self.leafnode:
            return self.sample_bounds()
        else:
            lp = self.lower.max_leaf_progress
            gp = self.greater.max_leaf_progress
            maxp = max(lp, gp)
            # no point sampling either child node because they're not sampleable
            if self.lower.can_sample is False and self.greater.can_sample is False:
                print("Cannot sample lower or greater child, sample current node instead")
                return self.sample_bounds()
            # if the lower child isn't sampleable, sample greater instead
            elif self.lower.can_sample is False:
                print("Cannot sample lower child")
                maxp = gp
            # alternatively, if the greater child isn't sampleable then sample lower instead
            elif self.greater.can_sample is False:
                print("Cannot sample greater child")
                maxp = lp
            if self.sampling_mode['multiscale']:                
                tp = self.progress        
                if tp > maxp:
                    return self.sample_bounds()
            if gp == maxp:
                sampling_mode = copy.deepcopy(self.sampling_mode)
                sampling_mode['mode'] = 'greedy'
                return self.greater.sample(sampling_mode=sampling_mode)
            else:
                sampling_mode = copy.deepcopy(self.sampling_mode)
                sampling_mode['mode'] = 'greedy'
                return self.lower.sample(sampling_mode=sampling_mode)
        
        
    def sample_epsilon_greedy(self, epsilon=0.1):
        """
        Sample a point in the leaf with the max potential learning progress with probability (1-eps) and a random leaf with probability (eps).
        
        Parameters
        ----------
        epsilon : float 
            
        """
        if epsilon > np.random.random():
            sampling_mode = copy.deepcopy(self.sampling_mode)  # This was updating the class instance because reference
            sampling_mode['mode'] = 'random'
            self.emit('sample', 'sampling random')
            return self.sample(sampling_mode=sampling_mode)
        else:
            sampling_mode = copy.deepcopy(self.sampling_mode)
            sampling_mode['mode'] = 'greedy'
            return self.sample(sampling_mode=sampling_mode)
        
        
    def sample_softmax(self, temperature=1.):
        """
        Sample leaves with probabilities progress*volume and a softmax exploration (with a temperature parameter).

        volume is the product of the bounds of exploration for that tree / subtree.

        Sampling always looks to maximize learning progress

        Parameters
        ----------
        temperature : float 
        
        """
        if self.leafnode:
            return self.sample_bounds() # random sample of bounds
        else:
            if self.sampling_mode['multiscale']:
                nodes = self.get_nodes()
            else:
                nodes = self.get_leaves()
                
            if  self.sampling_mode['volume']:
                progresses = np.array([node.progress*node.volume for node in nodes]) #by volume
            else:
                progresses = np.array([node.progress for node in nodes])
                
            progress_max = max(progresses)
            probas = np.exp(progresses / (progress_max*temperature))
            probas = probas / np.sum(probas)
            
            if np.isnan(np.sum(probas)): # if progress_max = 0 or nan value in dataset, eps-greedy sample
                return self.sample_epsilon_greedy()
            else:
                node = nodes[np.where(np.random.multinomial(1, probas) == 1)[0][0]]
                return node.sample_bounds()
        
            
    def sample(self, sampling_mode=None):
        """
        Sample a point in the leaf region with max competence progress (recursive).
        
        Parameters
        ----------
        sampling_mode : dict
            How to sample a point in the tree: {'multiscale':bool, 'mode':string, 'param':float}
            
        """
        if sampling_mode is None:
            sampling_mode = self.sampling_mode

        if sampling_mode['mode'] == 'random':
            return self.sample_random()
                
        elif sampling_mode['mode'] == 'greedy':
            return self.sample_greedy()
            
        elif sampling_mode['mode'] == 'epsilon_greedy':
            return self.sample_epsilon_greedy(sampling_mode['param'])
            
        elif sampling_mode['mode'] == 'softmax':
            return self.sample_softmax(sampling_mode['param'])
            
        else:
            raise NotImplementedError(sampling_mode)
            
            
    def progress_all(self):
        """
        Competence progress of the overall tree.
        
        """
        return self.progress_idxs(list(range(np.shape(self.get_data_x())[0] - self.progress_win_size,
                                        np.shape(self.get_data_x())[0])))
    
            
    def progress_idxs(self, idxs):
        """
        Competence progress on points of given indexes. (higher competence is better, lower prediction error the better)
        
        """
        if self.progress_measure == 'abs_deriv_cov':
            #  approach from explauto's discrete progress interest model
            if len(idxs) <= 1:
                return 0
            else:
                idxs = sorted(idxs)[- self.progress_win_size:]
                return abs(np.cov(list(zip(list(range(len(idxs))), self.get_data_c()[idxs])), rowvar=0)[0, 1])
            
        elif self.progress_measure == 'abs_deriv':
            # absolute difference between first and last points in the window
            if len(idxs) <= 1:
                return 0
            else:
                idxs = sorted(idxs)[- self.progress_win_size:]
                return np.abs(np.mean(np.diff(self.get_data_c()[idxs], axis=0)))

        elif self.progress_measure == 'abs_deriv_smooth':
            # absolute difference between first and last half of the window
            if len(idxs) <= 1:
                return 0
            else:
                idxs = sorted(idxs)[- self.progress_win_size:]
                idxs_competencies = self.get_data_c()[idxs]
                n_competencies = len(idxs_competencies)
                comp_beg = np.mean(idxs_competencies[:int(float(n_competencies)/2.)])
                comp_end = np.mean(idxs_competencies[int(float(n_competencies)/2.):])
                return np.abs(comp_end - comp_beg)
            
        elif self.progress_measure == 'bounded_smooth':
            # absolute difference between first and last half of the window
            if len(idxs) <= 1:
                return 0
            else:
                idxs = sorted(idxs)[- self.progress_win_size:]
                idxs_competencies = self.get_data_c()[idxs]
                n_competencies = len(idxs_competencies)
                comp_beg = np.mean(idxs_competencies[:int(float(n_competencies)/2.)])
                comp_end = np.mean(idxs_competencies[int(float(n_competencies)/2.):])
                diff = comp_end - comp_beg
                return (diff + 1) / 4

        else:
            raise NotImplementedError(self.progress_measure)
        
        
    def update_progress(self):
        """
        Update progress of sub-trees (not recursive).
        
        """
        self.progress = self.progress_idxs(self.idxs)
            
    
    def update_max_progress(self):
        """
        Compute progress of tree and max progress of sub-trees (not recursive).
        
        """
        self.update_progress()
        if self.leafnode:
            self.max_leaf_progress = self.progress
        else:
            self.max_leaf_progress = max(self.lower.max_leaf_progress, self.greater.max_leaf_progress)
            
        
    def add(self, idx):
        """
        Add an index to the tree (recursive).
        
        """
        if self.leafnode and self.n_children >= self.max_points_per_region and self.max_depth > 0:
            self.split()
        self.idxs.append(idx)
        if self.leafnode:
            leaf_point_was_added_to = self
        else:
            if self.get_data_x()[idx, self.split_dim] >= self.split_value:
                leaf_point_was_added_to = self.greater.add(idx)  # recurse add until it gets to a leaf node to add
            else:
                leaf_point_was_added_to = self.lower.add(idx)  # recurse add until it gets to a leaf node to add
        self.update_max_progress()
        self.n_children = self.n_children + 1
        return leaf_point_was_added_to # return leaf on which the point has been added
    
    
    def split(self):
        """
        Split the leaf node.
        
        """
        print("splitting")
        # self.emit("split", f"Splitting: {self.split_mode}") # comment out, was erroring in wip deletion notebook?
        if self.split_mode == 'random':
            # Split randomly between min and max of node's points on split dimension
            split_dim_data = self.get_data_x()[self.idxs, self.split_dim] # data on split dim
            split_min = min(split_dim_data)
            split_max = max(split_dim_data)
            # split_value = split_min + np.random.rand() * (split_max - split_min)
            split_value = split_min + self.region_deletion_rng.random() * (split_max - split_min)

        elif self.split_mode == 'median':
            # Split on median (which fall on the middle of two points for even max_points_per_region) 
            # of node's points on split dimension
            split_dim_data = self.get_data_x()[self.idxs, self.split_dim] # data on split dim
            split_value = np.median(split_dim_data)
            
        elif self.split_mode == 'middle':
            # Split on the middle of the region: might cause empty leaf
            split_dim_data = self.get_data_x()[self.idxs, self.split_dim] # data on split dim
            split_value = (self.bounds_x[0, self.split_dim] + self.bounds_x[1, self.split_dim]) / 2
            
        elif self.split_mode == 'best_interest_diff': 
            # See Baranes2012: Active Learning of Inverse Models with Intrinsically Motivated Goal Exploration in Robots
            #   - if strictly more than self.max_points_per_region points: chooses between self.max_points_per_region points random split values
            # the one that maximizes card(lower)*card(greater)* progress difference between the two
            #   - if equal or lower than self.max_points_per_region points: chooses between splits at the middle of each pair of consecutive points,
            # the one that maximizes card(lower)*card(greater)* progress difference between the two
            split_dim_data = self.get_data_x()[self.idxs, self.split_dim] # data on split dim
            split_min = min(split_dim_data)
            split_max = max(split_dim_data)
                        
            if len(self.idxs) > self.max_points_per_region:
                m = self.max_points_per_region # Constant that might be tuned: number of random split values to choose between
                # rand_splits = split_min + np.random.rand(m) * (split_max - split_min)
                rand_splits = split_min + self.region_deletion_rng.random(m) * (split_max - split_min)
                splits_fitness = np.zeros(m)
                for i in range(m):
                    lower_idx = list(np.array(self.idxs)[np.nonzero(split_dim_data <= rand_splits[i])[0]])
                    greater_idx = list(np.array(self.idxs)[np.nonzero(split_dim_data > rand_splits[i])[0]])
                    splits_fitness[i] = len(lower_idx) * len(greater_idx) * abs(self.progress_idxs(lower_idx) - 
                                                                               self.progress_idxs(greater_idx))
                split_value = rand_splits[np.argmax(splits_fitness)]
                
            else: # len(idxs) is same as max_points_per_region (or lower, but I don't see how we'd get in that state)

                m = self.max_points_per_region - 1
                splits = (np.sort(split_dim_data)[0:-1] + np.sort(split_dim_data)[1:]) / 2
                splits_fitness = np.zeros(m)
                for i in range(m):
                    lower_idx = list(np.array(self.idxs)[np.nonzero(split_dim_data <= splits[i])[0]])
                    greater_idx = list(np.array(self.idxs)[np.nonzero(split_dim_data > splits[i])[0]])
                    splits_fitness[i] = len(lower_idx) * len(greater_idx) * abs(self.progress_idxs(lower_idx) - 
                                                                               self.progress_idxs(greater_idx))
                split_value = splits[np.argmax(splits_fitness)]
        elif self.split_mode == 'variance_of_cos_sim':
            # split so variance of cos sim is maximal on either side. This will encourage splitting  "concepts" in space.
            # (cos sim of each half should be 1)
            split_dim_data = self.get_data_x()[self.idxs, self.split_dim] # data on split dim
            split_min = min(split_dim_data)
            split_max = max(split_dim_data)
            m = (len(split_dim_data) - 1)
            print(f"Trying out {m} splits") # need to move from max points per region to the region # because of the reflection re-splits
            # m = self.max_points_per_region - 1  # Constant that might be tuned: number of random split values to choose between
            # rand_splits = split_min + np.random.rand(m) * (split_max - split_min) # array of random vals above split min
            splits = (np.sort(split_dim_data)[0:-1] + np.sort(split_dim_data)[1:]) / 2
            splits_fitness = np.zeros(m)
            for i in range(m):
                lower_idx = list(np.array(self.idxs)[np.nonzero(split_dim_data <= splits[i])[0]])
                greater_idx = list(np.array(self.idxs)[np.nonzero(split_dim_data > splits[i])[0]])
                lower_idx_sensory = self.get_data_y()[lower_idx]
                greater_idx_sensory = self.get_data_y()[greater_idx]
                # calc cos sim of all sensori in lower index
                # TODO: double check this func is working how I want
                lower_cos_sims_variance = self.calc_tree_variance_of_cos_sims(lower_idx_sensory)
                greater_cos_sims_variance = self.calc_tree_variance_of_cos_sims(greater_idx_sensory)
                # splits_fitness[i] = len(lower_idx) * len(greater_idx) * abs(lower_cos_sims_variance -
                #                                                             greater_cos_sims_variance)

                splits_fitness[i] = len(lower_idx) * len(greater_idx) * (1 / (lower_cos_sims_variance + greater_cos_sims_variance))  # penalize large variance by dividing by sum. Multiply by len of each list to maximize more even splits
            split_value = splits[np.argmax(splits_fitness)]

        else:
            raise NotImplementedError

        # self.emit("split", f"Split dimension: {self.split_dim}, value: {split_value}") # todo: uncomment - it's angry in test delete nb
    
        lower_idx = list(np.array(self.idxs)[np.nonzero(split_dim_data <= split_value)[0]])
        greater_idx = list(np.array(self.idxs)[np.nonzero(split_dim_data > split_value)[0]])

        self.leafnode = False
        self.split_value = split_value

        # mod split dim+1, num of features(cols) in x. Each iteration will update to next dimension (or in case of 2 dims, just toggle between them)
        split_dim = np.mod(self.split_dim + 1, np.shape(self.get_data_x())[1])
        
        l_bounds_x = np.array(self.bounds_x)
        l_bounds_x[1, self.split_dim] = split_value
        
        g_bounds_x = np.array(self.bounds_x)
        g_bounds_x[0, self.split_dim] = split_value
        self.lower = Tree(self.get_data_x,
                          l_bounds_x,
                          self.get_data_y,
                          self.get_data_flow_uuid,
                          self.get_data_c,
                          self.get_data_nav_memory_map,
                          self.max_points_per_region,
                          self.max_depth - 1,
                          self.split_mode,
                          self.progress_win_size,
                          self.progress_measure,
                          self.sampling_mode,
                          idxs = lower_idx,
                          split_dim = split_dim,
                          region_deletion_rng=self.region_deletion_rng)
        
        self.greater = Tree(self.get_data_x,
                            g_bounds_x,
                            self.get_data_y,
                            self.get_data_flow_uuid,
                            self.get_data_c,
                            self.get_data_nav_memory_map,
                            self.max_points_per_region,
                            self.max_depth - 1,
                            self.split_mode,
                            self.progress_win_size,
                            self.progress_measure,
                            self.sampling_mode,
                            idxs = greater_idx,
                            split_dim = split_dim,
                            region_deletion_rng=self.region_deletion_rng)

    def calc_tree_variance_of_cos_sims(self, tree_sensory):
        sensory_combinations_idxs = list(combinations(range(len(tree_sensory)), 2))
        cos_sims = []
        for combo_idx_a, combo_idx_b in sensory_combinations_idxs:
            cos_sims.append(cosine_similarity([tree_sensory[combo_idx_a]], [tree_sensory[combo_idx_b]]).flatten()[0])
        cos_sims_variance = 100 if len(cos_sims) == 0 else np.var(cos_sims)
        return cos_sims_variance

    # Adapted from scipy.spatial.kdtree 
    def __query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf):

        side_distances = np.maximum(0,np.maximum(x-self.bounds_x[1],self.bounds_x[0]-x))
        if p != np.inf:
            side_distances **= p
            min_distance = np.sum(side_distances)
        else:
            min_distance = np.amax(side_distances)

        # priority queue for chasing nodes
        # entries are:
        #  minimum distance between the cell and the target
        #  distances between the nearest side of the cell and the target
        #  the head node of the cell
        q = [(min_distance,
              tuple(side_distances),
              self)]
        # priority queue for the nearest neighbors
        # furthest known neighbor first
        # entries are (-distance**p, i)
        neighbors = []

        if eps == 0:
            epsfac = 1
        elif p == np.inf:
            epsfac = 1/(1+eps)
        else:
            epsfac = 1/(1+eps)**p

        if p != np.inf and distance_upper_bound != np.inf:
            distance_upper_bound = distance_upper_bound**p

        while q:
            min_distance, side_distances, node = heappop(q)
            if node.leafnode:
                # brute-force
                data = self.get_data_x()[node.idxs]
                ds = minkowski_distance_p(data,x[np.newaxis,:],p)
                for i in range(len(ds)):
                    if ds[i] < distance_upper_bound:
                        if len(neighbors) == k:
                            heappop(neighbors)
                        heappush(neighbors, (-ds[i], node.idxs[i]))
                        if len(neighbors) == k:
                            distance_upper_bound = -neighbors[0][0]
            else:
                # we don't push cells that are too far onto the queue at all,
                # but since the distance_upper_bound decreases, we might get
                # here even if the cell's too far
                if min_distance > distance_upper_bound*epsfac:
                    # since this is the nearest cell, we're done, bail out
                    break
                # compute minimum distances to the children and push them on
                if x[node.split_dim] < node.split_value:
                    near, far = node.lower, node.greater
                else:
                    near, far = node.greater, node.lower

                # near child is at the same distance as the current node
                heappush(q,(min_distance, side_distances, near))

                # far child is further by an amount depending only
                # on the split value
                sd = list(side_distances)
                if p == np.inf:
                    min_distance = max(min_distance, abs(node.split_value-x[node.split_dim]))
                elif p == 1:
                    sd[node.split_dim] = np.abs(node.split_value-x[node.split_dim])
                    min_distance = min_distance - side_distances[node.split_dim] + sd[node.split_dim]
                else:
                    sd[node.split_dim] = np.abs(node.split_value-x[node.split_dim])**p
                    min_distance = min_distance - side_distances[node.split_dim] + sd[node.split_dim]

                # far child might be too far, if so, don't bother pushing it
                if min_distance <= distance_upper_bound*epsfac:
                    heappush(q,(min_distance, tuple(sd), far))

        if p == np.inf:
            return sorted([(-d,i) for (d,i) in neighbors])
        else:
            return sorted([((-d)**(1./p),i) for (d,i) in neighbors])
        
        
    # Adapted from scipy.spatial.kdtree 
    def nn(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf):
        """
        Query the tree for nearest neighbors

        Parameters
        ----------
        x : array_like, last dimension self.m
            An array of points to query.
        k : integer
            The number of nearest neighbors to return.
        eps : nonnegative float
            Return approximate nearest neighbors; the kth returned value
            is guaranteed to be no further than (1+eps) times the
            distance to the real kth nearest neighbor.
        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use.
            1 is the sum-of-absolute-values "Manhattan" distance
            2 is the usual Euclidean distance
            infinity is the maximum-coordinate-difference distance
        distance_upper_bound : nonnegative float
            Return only neighbors within this distance. This is used to prune
            tree searches, so if you are doing a series of nearest-neighbor
            queries, it may help to supply the distance to the nearest neighbor
            of the most recent point.

        Returns
        -------
        d : float or array of floats
            The distances to the nearest neighbors.
            If x has shape tuple+(self.m,), then d has shape tuple if
            k is one, or tuple+(k,) if k is larger than one. Missing
            neighbors (e.g. when k > n or distance_upper_bound is
            given) are indicated with infinite distances.  If k is None,
            then d is an object array of shape tuple, containing lists
            of distances. In either case the hits are sorted by distance
            (nearest first).
        i : integer or array of integers
            The locations of the neighbors in self.data. i is the same
            shape as d.

        """
        self.n, self.m = np.shape(self.get_data_x())
        x = np.asarray(x)
        if np.shape(x)[-1] != self.m:
            raise ValueError("x must consist of vectors of length %d but has shape %s" % (self.m, np.shape(x)))
        if p < 1:
            raise ValueError("Only p-norms with 1<=p<=infinity permitted")
        retshape = np.shape(x)[:-1]
        if retshape != ():
            if k is None:
                dd = np.empty(retshape,dtype=np.object)
                ii = np.empty(retshape,dtype=np.object)
            elif k > 1:
                dd = np.empty(retshape+(k,),dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(retshape+(k,),dtype=np.int)
                ii.fill(self.n)
            elif k == 1:
                dd = np.empty(retshape,dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(retshape,dtype=np.int)
                ii.fill(self.n)
            else:
                raise ValueError("Requested %s nearest neighbors; acceptable numbers are integers greater than or equal to one, or None")
            for c in np.ndindex(retshape):
                hits = self.__query(x[c], k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound)
                if k is None:
                    dd[c] = [d for (d,i) in hits]
                    ii[c] = [i for (d,i) in hits]
                elif k > 1:
                    for j in range(len(hits)):
                        dd[c+(j,)], ii[c+(j,)] = hits[j]
                elif k == 1:
                    if len(hits) > 0:
                        dd[c], ii[c] = hits[0]
                    else:
                        dd[c] = np.inf
                        ii[c] = self.n
            return dd, ii
        else:
            hits = self.__query(x, k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound)
            if k is None:
                return [d for (d,i) in hits], [i for (d,i) in hits]
            elif k == 1:
                if len(hits) > 0:
                    return hits[0]
                else:
                    return np.inf, self.n
            elif k > 1:
                dd = np.empty(k,dtype=float)
                dd.fill(np.inf)
                ii = np.empty(k,dtype=int)
                ii.fill(self.n)
                for j in range(len(hits)):
                    dd[j], ii[j] = hits[j]
                return dd, ii
            else:
                raise ValueError("Requested %s nearest neighbors; acceptable numbers are integers greater than or equal to one, or None")
         
                
    def fold_up(self, f_inter, f_leaf):
        """
        Apply recursively the function f_inter from leaves to root, begining with function f_leaf on leaves.
        
        """
        return f_leaf(self) if self.leafnode else f_inter(self.lower.fold_up(f_inter, f_leaf),
                                                          self.greater.fold_up(f_inter, f_leaf))


    def fold_up_sampleable(self, f_inter, f_leaf):
        """
        Apply recursively the function f_inter from leaves to root, begining with function f_leaf on leaves.

        """
        return f_leaf(self) if self.leafnode and self.can_sample else f_inter(self.lower.fold_up(f_inter, f_leaf),
                                                          self.greater.fold_up(f_inter, f_leaf))


    def competence_measure(self, target, reached):
        # return competence_exp(target, reached, 0, 10)
        return prediction_error_cos_dist_exp(target, reached)

    def plot(self, ax=None, ax2=None, scatter=True, grid=True, progress_colors=True, progress_max=1., depth=30, plot_dims=[0,1], legend_artists=None):
        """
        Plot a projection on 2D of the Tree.
        
        Parameters
        ----------
        ax : plt axis
        ax2: radial plot axis (todo: this could be implemented better)
        scatter : bool
            If the points are ploted
        grid : bool
            If the leaves' bounds are ploted
        progress_colors : bool
            If rectangles are filled with colors based on progress 
        progress_max : float
            Max progress on color scale (will be ploted as 1.)
        depth : int
            Max depth of the ploted nodes
        plot_dims : list
            List of the 2 dimensions to project tree on
        
        """
        if ax is not None:
            ax.clear()
            # cat_path = './retico/misc/cat_icon.png'
            # eleph_path = './retico/misc/elephant_icon.png'
            if grid:
                self.plot_grid(ax, progress_colors, progress_max, depth, plot_dims, legend_artists=legend_artists)
                if len(plot_dims) == 2: # TODO Catherine: Could we support marking known object locations on the 3d grid?
                    self.add_plot_objs(ax, "grid")
            if scatter and self.get_data_x() is not None:
                self.plot_scatter(ax=ax, plot_dims=plot_dims)
        if ax2 is not None:
            ax2.clear()
            self.add_plot_objs(ax2, "radial")
            self.plot_scatter_radians(ax=ax2)


    def add_plot_objs(self, ax, plot_type):

        max_forward_linear_travel = self.bounds_x[1][1]
        max_reverse_linear_travel = self.bounds_x[0][1]
        max_rotation_degree = self.bounds_x[1][0]
        min_rotation_degree = self.bounds_x[0][0]
        cozmo_fov = 56.5  # self.robot.camera.config.fov_x says cozmo horiz fov is 56.53 degrees

        if plot_type == "grid":
            for plot_obj in self.plot_objects:

                # The obj is straddling the x axis, so the polygon patch needs to occur on both ends of the plot meaning two patches are required
                if (plot_obj.rightmost_angle_from_0 > 0 and plot_obj.leftmost_angle_from_0 < 0) or (plot_obj.rightmost_angle_from_0 < 0 and plot_obj.leftmost_angle_from_0 > 0):
                    ax.add_patch(Polygon([
                        [(plot_obj.rightmost_angle_from_0 - (cozmo_fov/4)), max_forward_linear_travel], # minimum rotation to see right of obj with maximum forward linear movement
                        [(plot_obj.rightmost_angle_from_0 - (cozmo_fov/2)), 0],  # minimum rotation to see right of obj with no linear travel
                        [(plot_obj.rightmost_angle_from_0 - (cozmo_fov/1.5)), max_reverse_linear_travel], # minimum rotation to see right of obj with maximum reverse linear movement
                        [(180 + (cozmo_fov/1.5)), max_reverse_linear_travel], # minimum rotation to see left of obj with maximum reverse linear movement
                        [(180 +  (cozmo_fov/2)), 0], # minimum rotation to see left of obj with no linear travel
                        [(180 +  (cozmo_fov/4)), max_forward_linear_travel], # minimum rotation to see left of obj with maximum forward linear movement
                    ], fill=False,  edgecolor='#8aeb3f', alpha=0.3, hatch='xxx'))

                    # the positive side will plot fine. The negative we need to adjust
                    # the leftmost would be the negative angle (if cozmo is facing the negative x axis).
                    ax.add_patch(Polygon([
                        [(plot_obj.leftmost_angle_from_0 + (cozmo_fov/4)), max_forward_linear_travel], # minimum rotation to see right of obj with maximum forward linear movement
                        [(plot_obj.leftmost_angle_from_0 + (cozmo_fov/2)), 0],  # minimum rotation to see right of obj with no linear travel
                        [(plot_obj.leftmost_angle_from_0 + (cozmo_fov/1.5)), max_reverse_linear_travel], # minimum rotation to see right of obj with maximum reverse linear movement
                        [(-180 - (cozmo_fov/1.5)), max_reverse_linear_travel], # minimum rotation to see left of obj with maximum reverse linear movement
                        [(-180 - (cozmo_fov/2)), 0], # minimum rotation to see left of obj with no linear travel
                        [(-180 - (cozmo_fov/4)), max_forward_linear_travel], # minimum rotation to see left of obj with maximum forward linear movement
                    ], fill=False,  edgecolor='#8aeb3f', alpha=0.3, hatch='xxx'))

                    obj_ab_left = AnnotationBbox(OffsetImage(plot_obj.grid_image, resample=True, zoom=0.015), (plot_obj.leftmost_angle_from_0, max_forward_linear_travel), box_alignment=(0.5, -0.15), frameon=False)
                    ax.add_artist(obj_ab_left)
                    obj_ab_right = AnnotationBbox(OffsetImage(plot_obj.grid_image, resample=True, zoom=0.015), (plot_obj.rightmost_angle_from_0, max_forward_linear_travel), box_alignment=(0.5, -0.15), frameon=False)
                    ax.add_artist(obj_ab_right)

                else:
                    ax.add_patch(Polygon([
                        [(plot_obj.rightmost_angle_from_0 - (cozmo_fov/4)), max_forward_linear_travel], # minimum rotation to see right of obj with maximum forward linear movement
                        [(plot_obj.rightmost_angle_from_0 - (cozmo_fov/2)), 0],  # minimum rotation to see right of obj with no linear travel
                        [(plot_obj.rightmost_angle_from_0 - (cozmo_fov/1.5)), max_reverse_linear_travel], # minimum rotation to see right of obj with maximum reverse linear movement
                        [(plot_obj.leftmost_angle_from_0 + (cozmo_fov/1.5)), max_reverse_linear_travel], # minimum rotation to see left of obj with maximum reverse linear movement
                        [(plot_obj.leftmost_angle_from_0 + (cozmo_fov/2)), 0], # minimum rotation to see left of obj with no linear travel
                        [(plot_obj.leftmost_angle_from_0 + (cozmo_fov/4)), max_forward_linear_travel], # minimum rotation to see left of obj with maximum forward linear movement
                    ], fill=False,  edgecolor='#8aeb3f', alpha=0.3, hatch='xxx'))

                    obj_ab = AnnotationBbox(OffsetImage(plot_obj.grid_image, resample=True, zoom=0.015), (plot_obj.angle_from_0_avg, max_forward_linear_travel), box_alignment=(0.5, -0.15), frameon=False)
                    ax.add_artist(obj_ab)

            # ax.add_patch(Polygon([[112, -10], [140, -80], [168, -10], [140, 80]], facecolor="green", alpha=0.5))
            # ax.add_patch(Polygon([[2, -10], [30, -80], [58, -10], [30, 80]], facecolor="green", alpha=0.5))
            ax.set_xlim((-180,180))
            ax.set_ylim((max_reverse_linear_travel, max_forward_linear_travel))

        elif plot_type == "radial":
            ax.patch.set_facecolor('#c9e6c8') # pale green. to better show off old (white) points
            # mark off where points won't be selected because the recentering cube
            # NOTE: no longer needed with the cube elephant object
            # ax.add_artist(Wedge((.5,.5), 0.52, max_rotation_degree, min_rotation_degree, width=0.55, transform=ax.transAxes, color='white'))

            for plot_obj in self.plot_objects:

                ax.add_patch(Polygon([
                    [np.deg2rad(plot_obj.rightmost_angle_from_0 - (cozmo_fov/4)), max_forward_linear_travel], # minimum rotation to see right of obj with maximum forward linear movement
                    [np.deg2rad(plot_obj.rightmost_angle_from_0 - (cozmo_fov/2)), 0],  # minimum rotation to see right of obj with no linear travel
                    [np.deg2rad(plot_obj.rightmost_angle_from_0 - (cozmo_fov/1.5)), max_reverse_linear_travel], # minimum rotation to see right of obj with maximum reverse linear movement
                    [np.deg2rad(plot_obj.leftmost_angle_from_0 + (cozmo_fov/1.5)), max_reverse_linear_travel], # minimum rotation to see left of obj with maximum reverse linear movement
                    [np.deg2rad(plot_obj.leftmost_angle_from_0 + (cozmo_fov/2)), 0], # minimum rotation to see left of obj with no linear travel
                    [np.deg2rad(plot_obj.leftmost_angle_from_0 + (cozmo_fov/4)), max_forward_linear_travel], # minimum rotation to see left of obj with maximum forward linear movement
                    [np.deg2rad(plot_obj.leftmost_angle_from_0), 1000], # dummy point to fill in space (only necessary for polar plot)
                ], facecolor="green", alpha=0.5))

                # find box icon bounding box alignment for placement around the polar plot
                # https://math.stackexchange.com/questions/2740317/simple-way-to-find-position-on-square-given-angle-at-center
                bounding_box_angle = plot_obj.angle_from_0_avg + 180 # need opposite of intended angle bcz that is where we want the bounding box to "glue" to
                magic_number = bounding_box_angle - 90 * round(bounding_box_angle/90, 0) # see stackoverflow post above for this equation
                bounding_box_radius = 0.5 # per matplotlib docs the box ranges from 0,0 to 1,1
                x = bounding_box_radius * cos(radians(bounding_box_angle))/cos(radians(magic_number)) + bounding_box_radius
                y = bounding_box_radius * sin(radians(bounding_box_angle))/cos(radians(magic_number)) + bounding_box_radius
                obj_ab = AnnotationBbox(OffsetImage(plot_obj.polar_image, resample=True, zoom=0.015), (plot_obj.angle_from_0_avg * (np.pi/180), max_forward_linear_travel), box_alignment=(x,y), frameon=False)
                ax.add_artist(obj_ab)


    def plot_scatter(self, ax, plot_dims=[0,1]):

        # plot points on figure
        if np.shape(self.get_data_x())[0] <= 5000:
            if len(plot_dims) == 2:
                ax.set_xlabel("Degree of Rotation")
                ax.set_ylabel("mm Linear Travel (Post Rotation)")
                ax.scatter(self.get_data_x()[:,plot_dims[0]], self.get_data_x()[:,plot_dims[1]], color = 'snow')
            elif len(plot_dims) == 3:
                ax.set_xlabel("mm Travel along x Axis")
                ax.set_ylabel("mm Travel along y Axis")
                ax.set_zlabel("Degree of Rotation")
                ax.scatter(self.get_data_x()[:,0], self.get_data_x()[:,1], self.get_data_x()[:,2], color = 'black')



        ax.set_title(f'Action/Perception Turn Count: {len(self.get_data_x())}', loc='left', pad=30)

    def plot_scatter_radians(self, ax, plot_dims=[0,1]):
        # ax.patch.set_facecolor('snow')
        # ax.patch.set_facecolor('gainsboro')
        ax.patch.set_facecolor('#c9e6c8') # pale green. to better show off old (white) points


        # cozmo_fov = 56  # self.robot.camera.config.fov_x says cozmo horiz fov is 56.53 degrees
        # max_forward_linear_travel = 80
        # max_reverse_linear_travel = -80
        #

        # ax.add_patch(Polygon([[55 * (np.pi/180), 0], [0 * (np.pi/180), -80], [5 * (np.pi/180), 0], [30 * (np.pi/180), 80]], facecolor="green", alpha=0.3))
        # ax.add_patch(Polygon([[155 * (np.pi/180), 0],  [0 * (np.pi/180), -80], [115 * (np.pi/180), 0], [140 * (np.pi/180), 80]], facecolor="green", alpha=0.3))

        index_hue = (np.arange(len(self.get_data_x()))+1)/len(self.get_data_x())
        if np.shape(self.get_data_x())[0] <= 5000:
            ax.scatter(self.get_data_x()[:,plot_dims[0]] * np.pi/180, self.get_data_x()[:,plot_dims[1]], alpha=index_hue, color = 'black')

        ax.set_thetagrids(range(0, 360, 45), (0, 45, 90, 135, 180, -135, -90, -45))
        ax.set_rmax(80.0)
        ax.set_rmin(-80.0)
        ax.set_rlabel_position(-30)
        ax.set_title(f'Action/Perception Turn Count: {len(self.get_data_x())}', loc='left')

    def plot_grid(self, ax, progress_colors=True, progress_max=1., depth=10, plot_dims=[0,1], category_labels=None, legend_artists=None):
        debug = False
        if category_labels is None:
            category_labels = []
        if debug:
            print(f"depth {depth}")

        axis = depth % 3 # Cycle through x, y, z axes
        # print(f"axis {axis}")
        prog_min = 0.
        if self.leafnode or depth == 0:
            if debug:
                print("leafnode")
            mins = self.bounds_x[0,plot_dims]
            maxs = self.bounds_x[1,plot_dims]
            if debug:
                print(f"mins: {mins}")
                print(f"maxs: {maxs}")
                print(f"region line coordinates: {(mins[0], mins[1], mins[2]), (maxs[0], maxs[1], maxs[2])}")
            # Plot a rectangle in 2D space
            if len(plot_dims) == 2:
                if progress_colors:
                    prog_min = 0.
                    c = plt.cm.gnuplot((self.max_leaf_progress - prog_min) / (progress_max - prog_min)) if progress_max > prog_min else plt.cm.gnuplot(0)
                    ax.add_patch(plt.Rectangle(mins, maxs[0] - mins[0], maxs[1] - mins[1], facecolor=c,  edgecolor='white', alpha=0.7))
                    ax.annotate(len(category_labels), mins, color='#8dd17d', weight='bold', fontsize=15, ha='left', va='baseline')
                else:
                    ax.add_patch(plt.Rectangle(mins, maxs[0] - mins[0], maxs[1] - mins[1], fill=False))
            if len(plot_dims) == 3:
                # Plot the diagonal line of the min/max coordinates. We use this line + normal vector to determine the plane and then build the rectangle
                # print(f"\tline coordinates: {(split_value, mins[1], mins[2]), (split_value, maxs[1], maxs[2])}")
                # ax.plot([mins[0], maxs[0]], [mins[1], maxs[1]], [mins[2], maxs[2]], c='green')
                if axis == 0:
                    # print("AXIS 0")
                    # normal vector for splitting along X axis (used for calculating the plane that divides X axis)
                    if debug:
                        print("AXIS 0")
                    normal_vector = [1, 0, 0]
                    a, b, c = normal_vector

                    split_value = mins[1]
                    yy, zz = np.meshgrid([mins[1], maxs[1]], [mins[2], maxs[2]])
                    # have an ascending and descending set of y coordinates so they go in rectangle order (otherwise will produce an hourglass)
                    yy_asc = np.sort(yy)
                    yy_desc = -np.sort(-yy)

                    if debug:
                        print(f"\tyy:\n\t{yy}")
                        print(f"\tzz:\n\t{zz}")


                    p1  = np.array([mins[0], mins[1], mins[2]])

                    ## CALCULATE RECTANGLE 1
                    # Calculate z values using the plane equation: ax + by + cz + d = 0
                    # where (a, b, c) is the normal vector and d = -(ax0 + by0 + cz0) for a point (x0, y0, z0) on the plane
                    d = -np.dot(normal_vector, p1)
                    x1 = (-d  - b * yy - c * zz) / a
                    if debug:
                        # print(f"\ty1:\n\t{y1}")
                        print(f"\tRECTANGLE 1: \n{x1}\n{yy}\n{zz}")
                    r1_stacked = np.stack((x1, yy, zz),  axis=2)
                    if debug:
                        print(f"\t\tp1: {p1}, d1: {d}, x1: {x1}")

                    r1_coordinates = np.reshape(r1_stacked, (-1, 3))

                    if debug:
                        print(f"\t\tr1 stacked: \n{r1_stacked}")
                        print(f"\t\tr1 coordinates: \n{r1_coordinates}")
                    sorted_r1_coordinates = np.array(sorted(r1_coordinates.tolist()))
                    if debug:
                        print(f"\t\tsorted r1 coordinates: {sorted_r1_coordinates}")

                    ## CALCULATE RECTANGLE 2
                    p2  = np.array([maxs[0], maxs[1], maxs[2]])
                    d2 = -np.dot(normal_vector, p2)
                    x2 = (-d2  - b * yy - c * zz) / a
                    if debug:
                        print(f"\tRECTANGLE 2: \nx:{x2}\ny2:{yy}\nz:{zz}")
                        print(f"\t\tp2: {p2}, d2: {d2}, y2: {x2}")
                    r2_stacked = np.stack((x2, yy, zz),  axis=2)
                    r2_coordinates = np.reshape(r2_stacked, (-1, 3))

                    if debug:
                        print(f"\t\tr2 coordinates: \n{r2_coordinates}")

                    c = plt.cm.gnuplot((self.max_leaf_progress - prog_min) / (progress_max - prog_min)) if progress_max > prog_min else plt.cm.gnuplot(0)
                    split = ax.plot_surface(np.concatenate((x1, x2, x1, x2), axis=1), np.concatenate((yy_asc, yy_desc, yy_asc, yy_desc), axis=1), np.concatenate((zz, zz, zz, zz), axis=1), linewidth=2, alpha=.05, edgecolors=c, shade=False, color=c)
                    annotation = annotate3D(ax, s=str(len(category_labels)), xyz=mins, fontsize=10, xytext=(-3,3),
                               textcoords='offset points', ha='right',va='bottom')


                if axis == 1:
                    # print("AXIS 1")
                    # normal vector for splitting along Y axis (used for calculating the plane that divides Y axis)
                    normal_vector = [0, 1, 0]
                    a, b, c = normal_vector

                    split_value = mins[1]
                    xx, zz = np.meshgrid([mins[0], maxs[0]], [mins[2], maxs[2]])
                    # have an ascending and descending set of x coordinates so they go in rectangle order (otherwise will produce an hourglass)
                    xx_asc = np.sort(xx)
                    xx_desc = -np.sort(-xx)
                    if debug:
                        print(f"\txx:\n\t{xx}")
                        print(f"\tzz:\n\t{zz}")

                    p1  = np.array([mins[0], split_value, mins[2]])

                    ## CALCULATE RECTANGLE 1
                    # Calculate z values using the plane equation: ax + by + cz + d = 0
                    # where (a, b, c) is the normal vector and d = -(ax0 + by0 + cz0) for a point (x0, y0, z0) on the plane
                    d = -np.dot(normal_vector, p1)
                    y1 = (-d - a * xx - c * zz)/b
                    if debug:
                        print(f"\tp1: {p1}, d1: {d}, y1: {y1}")
                        # print(f"\ty1:\n\t{y1}")
                        print(f"\tRECTANGLE 1: \n{xx}\n{y1}\n{zz}")
                    r1_stacked = np.stack((xx, y1, zz),  axis=2)

                    r1_coordinates = np.reshape(r1_stacked, (-1, 3))
                    if debug:
                        print(f"\t\tr1 stacked: \n{r1_stacked}")
                        print(f"\t\tr1 coordinates: \n{r1_coordinates}")
                    sorted_r1_coordinates = np.array(sorted(r1_coordinates.tolist()))
                    if debug:
                        print(f"\t\tsorted r1 coordinates: {sorted_r1_coordinates}")

                    ## CALCULATE RECTANGLE 2
                    p2  = np.array([maxs[0], maxs[1], maxs[2]])
                    d2 = -np.dot(normal_vector, p2)
                    y2 = (-d2 - a * xx - c * zz)/b
                    if debug:
                        # print(f"\ty2:\n\t{y2}")
                        print(f"\tRECTANGLE 2: \nx:{xx}\ny2:{y2}\nz:{zz}")
                        print(f"\t\tp2: {p2}, d2: {d2}, y2: {y2}")
                        # print(np.concatenate((xx, y1, zz),axis=1).T)
                    r2_stacked = np.stack((xx, y2, zz),  axis=2)
                    r2_coordinates = np.reshape(r2_stacked, (-1, 3))
                    if debug:
                        # print(f"r2 stacked: \n{r2_stacked}")
                        print(f"\t\tr2 coordinates: \n{r2_coordinates}")


                    c = plt.cm.gnuplot((self.max_leaf_progress - prog_min) / (progress_max - prog_min)) if progress_max > prog_min else plt.cm.gnuplot(0)
                    split = ax.plot_surface(np.concatenate((xx_asc, xx_desc, xx_asc, xx_desc), axis=1), np.concatenate((y1, y2, y1, y2), axis=1), np.concatenate((zz, zz, zz, zz), axis=1), linewidth=2, alpha=.05, edgecolors=c, shade=False, color=c)
                    annotation = annotate3D(ax, s=str(len(category_labels)), xyz=mins, fontsize=10, xytext=(-3,3),
                           textcoords='offset points', ha='right',va='bottom')


                if axis == 2:
                    # print("AXIS 2")
                    # normal vector for splitting along Z axis (used for calculating the plane that divides Z axis)
                    normal_vector = [0, 0, 1]
                    a, b, c = normal_vector

                    split_value = mins[1]
                    xx, yy = np.meshgrid([mins[0], maxs[0]], [mins[1], maxs[1]])
                    xx_asc = np.sort(xx)
                    xx_desc = -np.sort(-xx)
                    if debug:
                        print(f"\txx:\n\t{xx}")

                        print(f"\tyy:\n\t{yy}")

                    p1  = np.array([mins[0], mins[1], mins[2]])

                    ## CALCULATE RECTANGLE 1
                    # Calculate z values using the plane equation: ax + by + cz + d = 0
                    # where (a, b, c) is the normal vector and d = -(ax0 + by0 + cz0) for a point (x0, y0, z0) on the plane
                    d = -np.dot(normal_vector, p1)
                    z1 = (-d - a * xx - b * yy) / c

                    if debug:
                        # print(f"\ty1:\n\t{y1}")
                        print(f"RECTANGLE 1: \n{xx}\n{yy}\n{z1}")
                    r1_stacked = np.stack((xx, yy, z1),  axis=2)

                    r1_coordinates = np.reshape(r1_stacked, (-1, 3))
                    if debug:
                        print(f"r1 stacked: \n{r1_stacked}")
                        print(f"r1 coordinates: \n{r1_coordinates}")
                    sorted_r1_coordinates = np.array(sorted(r1_coordinates.tolist()))
                    if debug:
                        print(f"sorted r1 coordinates: {sorted_r1_coordinates}")

                    ## CALCULATE RECTANGLE 2
                    p2  = np.array([maxs[0], maxs[1], maxs[2]])
                    d2 = -np.dot(normal_vector, p2)
                    z2 = (-d2 - a * xx - b * yy) / c
                    if debug:
                        print(f"RECTANGLE 2: \nx:{xx}\ny2:{yy}\nz:{z2}")

                    r2_stacked = np.stack((xx, yy, z2),  axis=2)
                    r2_coordinates = np.reshape(r2_stacked, (-1, 3))
                    if debug:
                        print(f"r2 coordinates: \n{r2_coordinates}")

                    c = plt.cm.gnuplot((self.max_leaf_progress - prog_min) / (progress_max - prog_min)) if progress_max > prog_min else plt.cm.gnuplot(0)

                    split = ax.plot_surface(np.concatenate((xx_asc, xx_desc, xx_asc, xx_desc), axis=1), np.concatenate((yy, yy, yy, yy), axis=1), np.concatenate((z1, z2, z1, z2), axis=1), linewidth=2, alpha=.05, edgecolors=c, shade=False, color=c)
                    annotation = annotate3D(ax, s=str(len(category_labels)), xyz=mins, fontsize=10, xytext=(-3,3),
                               textcoords='offset points', ha='right',va='bottom')

                if legend_artists is not None:
                    legend_artists[str(len(category_labels))] = [split, annotation]

        else:
            if debug:
                print("not leaf")
            category_labels.append(len(category_labels))
            self.lower.plot_grid(ax, progress_colors, progress_max, depth - 1, plot_dims, category_labels, legend_artists)
            category_labels.append(len(category_labels))
            self.greater.plot_grid(ax, progress_colors, progress_max, depth - 1, plot_dims, category_labels, legend_artists)

# foal_plot_obj = PlotObject(image_path='./retico/misc/foal_icon.png', nose_x=-1, nose_y=4.5, tail_x=1.5, tail_y=4.5)
# goat_plot_obj = PlotObject(image_path='./retico/misc/goat_icon.png', nose_x=2.3, nose_y=-3, tail_x=0.5, tail_y=-4)

# cat_plot_obj = PlotObject(image_path='./retico/misc/cat_icon.png', nose_x=-8.5, nose_y=5.5, tail_x=-6.75, tail_y=6.5)
# elephant_plot_obj = PlotObject(image_path='./retico/misc/elephant_icon.png', nose_x=11, nose_y=3, tail_x=9, tail_y=7.25)
# elephant_plot_obj = PlotObject(image_path='./retico/misc/elephant_icon.png', nose_x=1.5, nose_y=-8, tail_x=-1.5, tail_y=-8)


# foal_plot_obj = PlotObject(image_path='./retico/misc/foal_icon.png', nose_x=-1, nose_y=-4.5, tail_x=-1.5, tail_y=-4.6)
# goat_plot_obj = PlotObject(image_path='./retico/misc/goat_icon.png', nose_x=0.5, nose_y=-3, tail_x=2.3, tail_y=-4)
# self.plot_objects = [cat_plot_obj, elephant_plot_obj, foal_plot_obj, goat_plot_obj]





cat_plot_obj = PlotObject(image_path=os.path.dirname(os.path.abspath(__file__)) + '/../utils/plot_icons/cat_icon.png', nose_x=4, nose_y=7.5, tail_x=6, tail_y=8.5)
cat_blob_plot_obj = PlotObject(image_path=os.path.dirname(os.path.abspath(__file__)) + '/../utils/plot_icons/blob_icon.png', nose_x=4, nose_y=7.5, tail_x=6, tail_y=8.5)
elephant_plot_obj = PlotObject(image_path=os.path.dirname(os.path.abspath(__file__)) + '/../utils/plot_icons/elephant_icon.png', nose_x=-12.5, nose_y=-2.5, tail_x=-12.5, tail_y=2.25)
elephant_blob_plot_obj = PlotObject(image_path=os.path.dirname(os.path.abspath(__file__)) + '/../utils/plot_icons/blob_icon.png', nose_x=-12.5, nose_y=-2.5, tail_x=-12.5, tail_y=2.25)


interest_models = {'tree': (InterestTree, {'default': {'max_points_per_region': 100,
                                                       'max_depth': 20,
                                                       'split_mode': 'best_interest_diff',
                                                       'competence_measure': competence_exp,
                                                       'progress_win_size': 50,
                                                       'progress_measure': 'abs_deriv_smooth',                                                     
                                                       'sampling_mode': {'mode':'softmax', 
                                                                         'param':0.2,
                                                                         'multiscale':False,
                                                                         'volume':True}},
                                           'cozmo': {'max_points_per_region': 30, # twenty seems good so far
                                                       'max_depth': 50,
                                                       'split_mode': 'variance_of_cos_sim',
                                                       'competence_measure': prediction_error_cos_dist_exp,
                                                       'progress_win_size': 10, # TODO try 15?
                                                       'progress_measure': 'steep_reverse_sigmoid_time_weighted',
                                                       'sampling_mode': {'mode':'epsilon_greedy',
                                                                         'param':0.1,
                                                                         'multiscale':False,
                                                                         'volume':True},
                                                     },
                                           'cozmo_binary_obj_detection': {'max_points_per_region': 30, # twenty seems good so far
                                                     'max_depth': 50,
                                                     'split_mode': 'best_interest_diff',
                                                     'competence_measure': competence_exp,
                                                     'progress_win_size': 10, # TODO try 15?
                                                     'progress_measure': 'abs_deriv_smooth',
                                                     'sampling_mode': {'mode':'epsilon_greedy',
                                                                       'param':0.1,
                                                                       'multiscale':False,
                                                                       'volume':True},
                                                    'plot_objects': [cat_blob_plot_obj, elephant_blob_plot_obj],
                                                    'region_deletion':False},
                                           'cozmo_clip': {'max_points_per_region': 30, # twenty seems good so far
                                                    'max_depth': 50,
                                                    'split_mode': 'best_interest_diff',
                                                    'competence_measure': competence_cos_dist_exp,
                                                    'progress_win_size': 10, # TODO try 15?
                                                    'progress_measure': 'abs_deriv_smooth',
                                                    'sampling_mode': {'mode':'epsilon_greedy',
                                                                      'param':0.1,
                                                                      'multiscale':False,
                                                                      'volume':True},
                                                    'plot_objects': [cat_plot_obj, elephant_plot_obj],
                                                    'region_deletion':False},
                                           'cozmo_clip_cos_sim_split': {'max_points_per_region': 10, #30 # twenty seems good so far
                                                     'max_depth': 50,
                                                     'split_mode': 'variance_of_cos_sim',
                                                     'competence_measure': competence_cos_dist_exp,
                                                     'progress_win_size': 10, # TODO try 15?
                                                     'progress_measure': 'abs_deriv_smooth',
                                                     'sampling_mode': {'mode':'epsilon_greedy',
                                                                       'param':0.1,
                                                                       'multiscale':False,
                                                                       'volume':True},
                                                    'plot_objects': [cat_plot_obj, elephant_plot_obj],
                                                    'region_deletion':False},
                                           'cozmo_clip_cos_sim_split_and_learning_prog': {'max_points_per_region': 5, #5 for developing! use 30 normally #30, # twenty seems good so far
                                                     'max_depth': 50,
                                                     'split_mode': 'variance_of_cos_sim',
                                                     'competence_measure': competence_cos_dist_exp,
                                                     'progress_win_size': 10, # TODO try 15?
                                                     'progress_measure': 'steep_reverse_sigmoid_time_weighted',
                                                     'sampling_mode': {'mode':'epsilon_greedy',
                                                                       'param':0.1,
                                                                       'multiscale':False,
                                                                       'volume':True},
                                                    'plot_objects': [cat_plot_obj, elephant_plot_obj],
                                                    'region_deletion':False},
                                           'cozmo_clip_cos_sim_split_random_sampling': {'max_points_per_region': 10, #30 # twenty seems good so far
                                                    'max_depth': 50,
                                                    'split_mode': 'variance_of_cos_sim',
                                                    'competence_measure': competence_cos_dist_exp,
                                                    'progress_win_size': 5, #10, # TODO try 15?
                                                    'progress_measure': 'abs_deriv_smooth',
                                                    'sampling_mode': {'mode':'random',
                                                                      'multiscale':False,
                                                                      'volume':False}, # Do not even weight random by volume, do true random
                                                    'plot_objects': [cat_plot_obj, elephant_plot_obj],
                                                    'region_deletion':False},
                                           'cozmo_clip_cos_sim_split_with_region_deletion': {'max_points_per_region': 10, #30 # twenty seems good so far
                                                    'max_depth': 50,
                                                    'split_mode': 'variance_of_cos_sim',
                                                    'competence_measure': competence_cos_dist_exp,
                                                    'progress_win_size': 5, #10, # TODO try 15?
                                                    'progress_measure': 'abs_deriv_smooth',
                                                    'sampling_mode': {'mode':'random',
                                                                      'multiscale':False,
                                                                      'volume':True}, # Do not even weight random by volume, do true random
                                                    'plot_objects': [cat_plot_obj, elephant_plot_obj],
                                                    'region_deletion':True},
                                           })}



