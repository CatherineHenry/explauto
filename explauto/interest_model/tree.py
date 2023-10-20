from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import copy

import time
import numpy as np
import matplotlib.pyplot as plt

from heapq import heappop, heappush

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Polygon
from scipy.spatial.kdtree import minkowski_distance_p

from ..utils.utils import rand_bounds
from ..utils.config import make_configuration
from .interest_model import InterestModel
from .competences import competence_exp, competence_dist, competence_cos_exp, cos_distance, competence_cos_log, prediction_error_cos_dist_exp
from ..utils.observer import Observable


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
                 sampling_mode):
        self.conf = conf
        self.bounds = self.conf.bounds[:, expl_dims]
        self.competence_measure = competence_measure
        
        if progress_win_size >= max_points_per_region:
            raise ValueError("WARNING: progress_win_size should be < max_points_per_region")
        
        self.data_x = None # list of target motor or sensory goals 'x'
        self.data_y = None # list of reached sensory effect
        self.data_c = None # list of competence measures

        self.tree = Tree(lambda:self.data_x, 
                         np.array(self.bounds, dtype=float),
                         lambda:self.data_y,
                         lambda:self.data_c, 
                         max_points_per_region=max_points_per_region, 
                         max_depth=max_depth,
                         split_mode=split_mode, 
                         progress_win_size=progress_win_size, 
                         progress_measure=progress_measure, 
                         sampling_mode=sampling_mode,
                         idxs=[])
        
        InterestModel.__init__(self, expl_dims)
        Observable.__init__(self)


    def sample(self):
        return self.tree.sample()
    
    def progress(self):
        return self.tree.progress
    
    def max_leaf_progress(self):
        return self.tree.max_leaf_progress
    
    def update(self, xy, ms, flow_uuid=None):
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

        cos_sim, cos_dist, bounded_cos_dist = self.competence_measure(xy, ms)
        self.emit(f"[{flow_uuid}] competence", f"[cos sim: {cos_sim}, cost dist: {cos_dist}] bounded cos distance between target and reached: {bounded_cos_dist}")
        if self.data_c is None:
            self.data_c = np.array([bounded_cos_dist]) # Either prediction error or competence error
        else:
            self.data_c = np.append(self.data_c, bounded_cos_dist)

        if self.data_y is None:
            self.data_y = np.array([ms[~np.isin(np.arange(len(ms)), self.expl_dims)]])
        else:
            self.data_y = np.append(self.data_y, np.array([ms[~np.isin(np.arange(len(ms)), self.expl_dims)]]), axis=0)
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
                 get_data_c, 
                 max_points_per_region, 
                 max_depth,
                 split_mode, 
                 progress_win_size, 
                 progress_measure, 
                 sampling_mode, 
                 idxs=None, 
                 split_dim=0):

        self.get_data_x = get_data_x
        self.bounds_x = np.array(bounds_x, dtype=np.float64)
        self.get_data_y = get_data_y
        self.get_data_c = get_data_c
        self.max_points_per_region = max_points_per_region
        self.max_depth = max_depth
        self.split_mode = split_mode
        self.progress_win_size = progress_win_size
        self.progress_measure = progress_measure
        self.sampling_mode = sampling_mode

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
        return self.fold_up(lambda n, fl, fg: [n] + fl + fg, lambda leaf: [leaf])
    
    
    def get_leaves(self):
        """
        Get the list of all leaves.
        
        """
        return self.fold_up(f_inter=lambda n, fl, fg: fl + fg, f_leaf =lambda leaf: [leaf])
    
    
    def depth(self):
        """
        Compute the depth of the tree (depth of a leaf=0).
        
        """
        return self.fold_up(lambda n, fl, fg: max(fl + 1, fg + 1), lambda leaf: 0)
    
    
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
        
        
    def sample_bounds(self):
        """
        Sample a point in the region of this node.
        
        """
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
                split_ratio = ((self.split_value - self.bounds_x[0,self.split_dim]) / 
                               (self.bounds_x[1,self.split_dim] - self.bounds_x[0,self.split_dim]))
                if split_ratio > np.random.random():
                    return self.lower.sample(sampling_mode={'mode':'random'})
                else:
                    return self.greater.sample(sampling_mode={'mode':'random'})
        else: 
            # Choose a leaf randomly
            return np.random.choice(self.get_leaves()).sample_bounds()
        
        
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
        Competence progress on points of given indexes. (lower competence the better)
        
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
        self.emit("split", f"Splitting: {self.split_mode}")
        if self.split_mode == 'random':
            # Split randomly between min and max of node's points on split dimension
            split_dim_data = self.get_data_x()[self.idxs, self.split_dim] # data on split dim
            split_min = min(split_dim_data)
            split_max = max(split_dim_data)
            split_value = split_min + np.random.rand() * (split_max - split_min)
            
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
                rand_splits = split_min + np.random.rand(m) * (split_max - split_min)
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
            m = self.max_points_per_region - 1  # Constant that might be tuned: number of random split values to choose between
            # rand_splits = split_min + np.random.rand(m) * (split_max - split_min) # array of random vals above split min
            splits = (np.sort(split_dim_data)[0:-1] + np.sort(split_dim_data)[1:]) / 2
            splits_fitness = np.zeros(m)
            for i in range(m):
                lower_idx = list(np.array(self.idxs)[np.nonzero(split_dim_data <= splits[i])[0]])
                greater_idx = list(np.array(self.idxs)[np.nonzero(split_dim_data > splits[i])[0]])
                lower_idx_sensory = self.get_data_y()[lower_idx]
                greater_idx_sensory = self.get_data_y()[greater_idx]
                # calc cos sim of all sensori in lower index
                lower_idx_combinations = list(combinations(range(len(lower_idx_sensory)), 2))
                lower_cos_sims = []
                for combo_idx_a, combo_idx_b in lower_idx_combinations:
                    lower_cos_sims.append(cosine_similarity([lower_idx_sensory[combo_idx_a]], [lower_idx_sensory[combo_idx_b]]).flatten()[0])
                lower_cos_sims_variance = 100 if len(lower_cos_sims) == 0 else np.var(lower_cos_sims)

                greater_idx_combinations = list(combinations(range(len(greater_idx_sensory)), 2))
                greater_cos_sims = []
                for combo_idx_a, combo_idx_b in greater_idx_combinations:
                    greater_cos_sims.append(cosine_similarity([greater_idx_sensory[combo_idx_a]], [greater_idx_sensory[combo_idx_b]]).flatten()[0])
                greater_cos_sims_variance = 100 if len(greater_cos_sims) == 0 else  np.var(greater_cos_sims)

                # splits_fitness[i] = len(lower_idx) * len(greater_idx) * abs(lower_cos_sims_variance -
                #                                                             greater_cos_sims_variance)

                splits_fitness[i] = len(lower_idx) * len(greater_idx) * (1 / (lower_cos_sims_variance + greater_cos_sims_variance))  # penalize large variance by dividing by sum. Multiply by len of each list to maximize more even splits
            split_value = splits[np.argmax(splits_fitness)]

        else:
            raise NotImplementedError

        self.emit("split", f"Split dimension: {self.split_dim}, value: {split_value}")
    
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
                         self.get_data_c, 
                         self.max_points_per_region, 
                         self.max_depth - 1,
                         self.split_mode, 
                         self.progress_win_size, 
                         self.progress_measure, 
                         self.sampling_mode, 
                         idxs = lower_idx, 
                         split_dim = split_dim)
        
        self.greater = Tree(self.get_data_x, 
                            g_bounds_x,
                            self.get_data_y,
                            self.get_data_c, 
                            self.max_points_per_region, 
                            self.max_depth - 1,
                            self.split_mode, 
                            self.progress_win_size, 
                            self.progress_measure, 
                            self.sampling_mode, 
                            idxs = greater_idx, 
                            split_dim = split_dim)
        
        
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
        return f_leaf(self) if self.leafnode else f_inter(self,
                                                          self.lower.fold_up(f_inter, f_leaf), 
                                                          self.greater.fold_up(f_inter, f_leaf))




    def plot(self, ax, ax2=None, scatter=True, grid=True, progress_colors=True, progress_max=1., depth=30, plot_dims=[0,1], plot_objects=None): #cat_path='./retico/misc/cat_icon.png', eleph_path = './retico/misc/elephant_icon.png'):
        """
        Plot a projection on 2D of the Tree.
        
        Parameters
        ----------
        ax : plt axis
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
        ax.clear()
        # cat_path = './retico/misc/cat_icon.png'
        # eleph_path = './retico/misc/elephant_icon.png'
        if grid or scatter:
            self.add_plot_objs(ax, plot_objects, "grid")
        if grid:
            self.plot_grid(ax, progress_colors, progress_max, depth, plot_dims)
        if scatter and self.get_data_x() is not None:
            self.plot_scatter(ax=ax, plot_dims=plot_dims)
        if ax2 is not None:
            ax2.clear()
            self.add_plot_objs(ax2, plot_objects, "radial")
            self.plot_scatter_radians(ax=ax2)


    def add_plot_objs(self, ax, plot_objects, plot_type):
        if plot_objects is None:
            plot_objects = []

        max_forward_linear_travel = 80
        max_reverse_linear_travel = -80

        cozmo_fov = 56  # self.robot.camera.config.fov_x says cozmo horiz fov is 56.53 degrees

        if plot_type == "grid":
            for plot_obj in plot_objects:

                ax.add_patch(Polygon([
                    [(plot_obj.rightmost_angle_from_0 - (cozmo_fov/4)), max_forward_linear_travel], # minimum rotation to see right of obj with maximum forward linear movement
                    [(plot_obj.rightmost_angle_from_0 - (cozmo_fov/2)), 0],  # minimum rotation to see right of obj with no linear travel
                    [(plot_obj.rightmost_angle_from_0 - (cozmo_fov/4)), max_reverse_linear_travel], # minimum rotation to see right of obj with maximum reverse linear movement
                    [(plot_obj.leftmost_angle_from_0 + (cozmo_fov/4)), max_reverse_linear_travel], # minimum rotation to see left of obj with maximum reverse linear movement
                    [(plot_obj.leftmost_angle_from_0 + (cozmo_fov/2)), 0], # minimum rotation to see left of obj with no linear travel
                    [(plot_obj.leftmost_angle_from_0 + (cozmo_fov/4)), max_forward_linear_travel], # minimum rotation to see left of obj with maximum forward linear movement
                ], fill=False,  edgecolor='#8aeb3f', alpha=0.3, hatch='xxx'))

                obj_ab = AnnotationBbox(OffsetImage(plot_obj.image, zoom=0.015), (plot_obj.angle_from_0_avg, max_forward_linear_travel), box_alignment=(0.5, -0.15), frameon=False)
                ax.add_artist(obj_ab)

            # ax.add_patch(Polygon([[112, -10], [140, -80], [168, -10], [140, 80]], facecolor="green", alpha=0.5))
            # ax.add_patch(Polygon([[2, -10], [30, -80], [58, -10], [30, 80]], facecolor="green", alpha=0.5))
            ax.set_xlim((-180,180))
            ax.set_ylim((max_reverse_linear_travel, max_forward_linear_travel))

        elif plot_type == "radial":

            for plot_obj in plot_objects:

                ax.add_patch(Polygon([
                    [(plot_obj.rightmost_angle_from_0 - (cozmo_fov/4))*(np.pi/180), max_forward_linear_travel], # minimum rotation to see right of obj with maximum forward linear movement
                    [(plot_obj.rightmost_angle_from_0 - (cozmo_fov/2))*(np.pi/180), 0],  # minimum rotation to see right of obj with no linear travel
                    [(plot_obj.rightmost_angle_from_0 - (cozmo_fov/4))*(np.pi/180), max_reverse_linear_travel], # minimum rotation to see right of obj with maximum reverse linear movement
                    [(plot_obj.leftmost_angle_from_0 + (cozmo_fov/4))*(np.pi/180), max_reverse_linear_travel], # minimum rotation to see left of obj with maximum reverse linear movement
                    [(plot_obj.leftmost_angle_from_0 + (cozmo_fov/2))*(np.pi/180), 0], # minimum rotation to see left of obj with no linear travel
                    [(plot_obj.leftmost_angle_from_0 + (cozmo_fov/4))*(np.pi/180), max_forward_linear_travel], # minimum rotation to see left of obj with maximum forward linear movement
                    [(plot_obj.leftmost_angle_from_0) * (np.pi/180), 1000], # dummy point to fill in space (only necessary for polar plot)
                ], facecolor="green", alpha=0.5))

                box_alignment_vert_direction = 4 if plot_obj.leftmost_angle_from_0 < 0 or plot_obj.rightmost_angle_from_0 < 0 else -1 # positive shifts image down, neg shifts image up
                obj_ab = AnnotationBbox(OffsetImage(plot_obj.image, zoom=0.015), (plot_obj.angle_from_0_avg * (np.pi/180), max_forward_linear_travel), box_alignment=(0.5, 0.15 * box_alignment_vert_direction), frameon=False)
                ax.add_artist(obj_ab)


    def plot_scatter(self, ax, plot_dims=[0,1]):

        ax.set_xlabel("degree of rotation")
        ax.set_ylabel("mm of linear travel (post rotation)")


        # plot points on figure
        if np.shape(self.get_data_x())[0] <= 5000:
            ax.scatter(self.get_data_x()[:,plot_dims[0]], self.get_data_x()[:,plot_dims[1]], color = 'snow')



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
        
    def plot_grid(self, ax, progress_colors=True, progress_max=1., depth=10, plot_dims=[0,1], category_labels=None):
        if category_labels is None:
            category_labels = []
        if self.leafnode or depth == 0:
        
            mins = self.bounds_x[0,plot_dims]
            maxs = self.bounds_x[1,plot_dims]
            
            if progress_colors:
                prog_min = 0.
                c = plt.cm.gnuplot((self.max_leaf_progress - prog_min) / (progress_max - prog_min)) if progress_max > prog_min else plt.cm.gnuplot(0)
                ax.add_patch(plt.Rectangle(mins, maxs[0] - mins[0], maxs[1] - mins[1], facecolor=c,  edgecolor='white', alpha=0.7))
                ax.annotate(len(category_labels), mins, color='#8dd17d', weight='bold', fontsize=15, ha='left', va='baseline')
            else:
                ax.add_patch(plt.Rectangle(mins, maxs[0] - mins[0], maxs[1] - mins[1], fill=False))
                    
        else:
            category_labels.append(len(category_labels))
            self.lower.plot_grid(ax, progress_colors, progress_max, depth - 1, plot_dims, category_labels)
            category_labels.append(len(category_labels))
            self.greater.plot_grid(ax, progress_colors, progress_max, depth - 1, plot_dims, category_labels)






interest_models = {'tree': (InterestTree, {'default': {'max_points_per_region': 100,
                                                       'max_depth': 20,
                                                       'split_mode': 'best_interest_diff',
                                                       'competence_measure': lambda target,reached : competence_exp(target, reached, 0., 10.),
                                                       'progress_win_size': 50,
                                                       'progress_measure': 'abs_deriv_smooth',                                                     
                                                       'sampling_mode': {'mode':'softmax', 
                                                                         'param':0.2,
                                                                         'multiscale':False,
                                                                         'volume':True}},
                                           'cozmo': {'max_points_per_region': 20, # twenty seems good so far
                                                       'max_depth': 50,
                                                       'split_mode': 'variance_of_cos_sim', # TODO: change split mode to cos sim?
                                                        # power 10 to give more weight to small differences
                                                       'competence_measure': lambda target,reached : prediction_error_cos_dist_exp(target, reached),
                                                       'progress_win_size': 8,
                                                       'progress_measure': 'abs_deriv_smooth',
                                                       'sampling_mode': {'mode':'epsilon_greedy',
                                                                         'param':0.1,
                                                                         'multiscale':False,
                                                                         'volume':True}}})}



