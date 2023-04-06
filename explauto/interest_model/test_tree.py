
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

from interest_model import interest_models
from interest_model.tree import Tree, InterestTree
from ..utils.utils import rand_bounds
from ..utils.config import make_configuration
from .interest_model import InterestModel
from .competences import competence_exp, competence_dist, competence_cos_exp, cos_distance, competence_cos_log, prediction_error_cos_dist_exp
from ..utils.observer import Observable



if __name__ == '__main__':
    # Tested from explauto/ with python -m explauto.interest_model.riac


    ######################################
    ########## TEST TREE #################
    ######################################

    if True:
        print("\n########## TEST TREE #################")
        n = 100000
        k = 2

        bounds = np.zeros((2, k))
        bounds[1,:] = 1

        data_x = rand_bounds(bounds, n)
        data_c = np.random.rand(n, 1)

        max_points_per_region = 5
        split_mode = 'median'
        progress_win_size = 10
        sampling_mode = interest_models['tree'][1]['default']['sampling_mode']

        #print get_data_x, get_data_c

        tree = Tree(lambda:data_x,
                    bounds,
                    lambda:data_c,
                    max_points_per_region,
                    20,
                    split_mode,
                    progress_win_size,
                    'abs_deriv',
                    sampling_mode,
                    list(range(n)))

        print("Sampling", tree.sample())
        print("Progress", tree.progress)
        tree.add(42)


        ####### FIND Neighrest Neighbors (might be useful)
        t = time.time()
        dist, idx = tree.nn([0.5, 0.5], k=20)
        print("Time to find neighrest neighbors:", time.time() - t)
        print(data_x[idx])


    ######################################
    ########## TEST InterestTree #########
    ######################################

    if True:
        print("\n########## TEST InterestTree #########")

        np.random.seed(1)

        max_points_per_region = 20
        split_mode = 'best_interest_diff'
        #split_mode = 'median'
        #split_mode = 'middle'

        # WARNING: progress_win_size has to be < than max_points_per_region.
        # If not, an improbably low competence will forever (in subtrees)
        # be taken into account in the computation of progress, leading to high progress
        # (if progress_measure='abs_deriv') and sampling forever in that region
        progress_win_size = 10
        sampling_mode = interest_models['tree'][1]['default']['sampling_mode']

        m_mins = [0, 0]
        m_maxs = [1, 1]
        s_mins = [3, 3]
        s_maxs = [4, 4]
        conf = make_configuration(m_mins, m_maxs, s_mins, s_maxs)

        expl_dims = [2, 3]

        riac = InterestTree(conf,
                            expl_dims,
                            max_points_per_region,
                            20,
                            split_mode,
                            competence_dist,
                            progress_win_size,
                            'abs_deriv',
                            sampling_mode)

        #print "Sample: ", riac.sample()

        # TEST UNIFORM RANDOM POINTS BATCH

        n = 1000
        xys = []
        mss = []
        for i in range(n):
            xys.append(rand_bounds(conf.bounds, 1)[0])
            mss.append(rand_bounds(conf.bounds, 1)[0])

        for i in range(n): # updated after for random seed purpose
            riac.update(xys[i], mss[i])

        fig1 = plt.figure()
        ax = fig1.add_subplot(111, aspect='equal')
        plt.xlim((riac.tree.bounds_x[0, 0], riac.tree.bounds_x[1, 0]))
        plt.ylim((riac.tree.bounds_x[0, 1], riac.tree.bounds_x[1, 1]))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('R-IAC tiling')
        riac.tree.plot(ax, True, True, True, riac.progress())

        print("Max nb of children:", riac.tree.fold_up(lambda n,fl,fg:max(fl,fg), lambda leaf:leaf.n_children))

        print("Max leaf progress: ", riac.max_leaf_progress())
        #         import matplotlib.colorbar as cbar
        #         cax, _ = cbar.make_axes(ax)
        #         cb = cbar.ColorbarBase(cax, cmap=plt.cm.jet)
        #         cb.set_label('Normalized Competence Progress')

        plt.show(block=False)


    ######################################
    ###### TEST PROGRESSING SAMPLING #####
    ######################################


    if True:
        print("\n###### TEST PROGRESSING SAMPLING #####")

        np.random.seed(1)

        max_points_per_region = 100
        progress_win_size = 50
        split_mode = 'best_interest_diff'
        #split_mode = 'median'
        #split_mode = 'middle'

        # WARNING: progress_win_size has to be < than max_points_per_region.
        # If not, an unprobably low competence will forever (in subtrees)
        # be taken into account in the computation of progress, leading to high progress
        # (if progress_measure='abs_deriv') and sampling forever in that region
        sampling_mode = interest_models['tree'][1]['default']['sampling_mode']

        m_mins = [0, 0]
        m_maxs = [1, 1]
        s_mins = [3, 3]
        s_maxs = [4, 4]
        conf = make_configuration(m_mins, m_maxs, s_mins, s_maxs)

        expl_dims = [2, 3]

        riac = InterestTree(conf,
                            expl_dims,
                            max_points_per_region,
                            20,
                            split_mode,
                            competence_dist,
                            progress_win_size,
                            'abs_deriv',
                            sampling_mode)

        n = 3000

        fig1 = plt.figure()
        ax = fig1.add_subplot(111, aspect='equal')
        ax.set_xlim((riac.tree.bounds_x[0, 0], riac.tree.bounds_x[1, 0]))
        ax.set_ylim((riac.tree.bounds_x[0, 1], riac.tree.bounds_x[1, 1]))
        riac.tree.plot(ax, True, True, True, riac.progress())

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('R-IAC tiling')

        #         import matplotlib.colorbar as cbar
        #         cax, _ = cbar.make_axes(ax)
        #         cb = cbar.ColorbarBase(cax, cmap=plt.cm.jet)
        #         cb.set_label('Normalized Competence Progress')

        plt.ion()
        plt.show()

        #         mng = plt.get_current_fig_manager()
        #         mng.resize(*mng.window.maxsize())


        for i in range(n):
            # SAMPLE A POINT, COMPUTE A SIMULATED REACHED POINT, ADD IT TO THE INTEREST MODEL
            xy = rand_bounds(conf.bounds, 1)[0]
            ms = rand_bounds(conf.bounds, 1)[0]
            sample = riac.tree.sample()
            #print "Sampled point: ", sample
            #print np.random.choice(riac.tree.get_leaves()).sample_bounds()
            xy[expl_dims] = sample


            # HERE we try to simulate a competence based on the quantity of exploration in the region,
            # with more progress in the middle of the map, no progress in the bottom left part,
            # random competences in the top left part.
            # Not sure how it can be interpreted
            # Need robotic setup for ecological testing

            leaf = riac.tree.pt2leaf(sample)
            density = leaf.density()
            #         if i > 0:
            #             print "comps", leaf.data_c[leaf.idxs]
            #print "Density:", density

            center = [3.5, 3.5]
            dist = np.linalg.norm(xy[expl_dims]- center)
            #print "dist", dist
            if sample[0] < 3.5 and sample[1] < 3.5:
                dist_reached = 10
            elif sample[0] < 3.5 and sample[1] > 3.5:
                dist_reached = 1. * np.random.random()
            else:
                if density < 5000:
                    dist_reached = 2. / ((density/5000. + 1) * (dist + 0.1))
                else:
                    dist_reached = 1. / (dist + 0.1)

            #ms[expl_dims] = np.random.normal(sample, eps)
            #print "dist_reached", dist_reached
            dist_reached = dist_reached# + np.random.random() * 0.001
            ms[expl_dims] = [sample[0]+dist_reached, sample[1]]
            #print "Number of leaves:", len(riac.tree.get_leaves())
            #print sample


            # ADD SAMPLE
            riac.update(xy, ms)

            # UPDATE PLOT
            if np.mod(i + 1, 100) == 0:
                print("Iteration:", i + 1, " Tree depth:", riac.tree.depth(), " Progress:", riac.progress(), "Max leaf progress", riac.max_leaf_progress())
                ax.clear()
                riac.tree.plot(ax, False, True, True, 10., 12)#riac.progress())
                plt.draw()
                plt.show()

        ax.clear()
        riac.tree.plot(ax, True, True, True, riac.progress(), 12)
        ax.set_xlim((riac.tree.bounds_x[0, 0], riac.tree.bounds_x[1, 0]))
        ax.set_ylim((riac.tree.bounds_x[0, 1], riac.tree.bounds_x[1, 1]))
        plt.draw()

        print("Sampling in max progress region: ", riac.tree.sample({'mode':'greedy', 'multiscale':True}))
        distances, idxs = riac.tree.nn(center, 10)
        print("Nearest Neighbors:", riac.data_x[idxs])

        plt.ioff()
    plt.show()



# import matplotlib.pyplot as plt
# fig1 = plt.figure()
# ax = fig1.add_subplot(111, aspect='equal')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Tree Plot')
# self.interest_model.tree.plot(ax)
# plt.show()