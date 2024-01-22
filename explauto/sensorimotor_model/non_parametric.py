
import numpy as np

from numpy import array

from ..exceptions import ExplautoBootstrapError
from .sensorimotor_model import SensorimotorModel
from .learner import Learner
from explauto.utils import bounds_min_max
from explauto.utils import rand_bounds


class NonParametric(SensorimotorModel):
    """ This class wraps the non-parametric forward and inverse models implemented by Fabien Benureau, in order to fit into the Explauto framework.
        Original code available at https://github.com/humm/models
        Adapted by Sebastien Forestier
    """
    def __init__(self, conf, sigma_explo_ratio=0.1, fwd='LWLR', inv='L-BFGS-B', **learner_kwargs):

        SensorimotorModel.__init__(self, conf)
        for attr in ['m_ndims', 's_ndims', 'm_dims', 's_dims', 'bounds', 'm_mins', 'm_maxs']:
            setattr(self, attr, getattr(conf, attr))

        self.sigma_expl = (conf.m_maxs - conf.m_mins) * float(sigma_explo_ratio)
        self.mode = 'explore'
        mfeats = tuple(range(self.m_ndims))
        sfeats = tuple(range(-self.s_ndims, 0))
        mbounds = tuple((self.bounds[0, d], self.bounds[1, d]) for d in range(self.m_ndims))

        self.model = Learner(mfeats, sfeats, mbounds, fwd, inv, **learner_kwargs)
        self.n_updates = 0  # number of points added to tree
        self.bootstrapped_s = False

    def infer(self, in_dims, out_dims, x):
        # I think `k` here is the number of neighbors used in weighting the nearest neighbor
        if self.n_updates < max(self.model.inv_model.fwd_model.k, self.model.inv_model.k):
            raise ExplautoBootstrapError

        if in_dims == self.m_dims and out_dims == self.s_dims:  # forward
            return array(self.model.predict_effect(tuple(x))) # predict the sensory effect of the motor action

        elif in_dims == self.s_dims and out_dims == self.m_dims:  # inverse
            if not self.bootstrapped_s:
                # If only one distinct point has been observed in the sensory space, then we output a random motor command
                return rand_bounds(np.array([self.m_mins,
                                             self.m_maxs]))[0]
            else:
                if self.mode == 'explore':
                    self.mean_explore = array(self.model.infer_order(tuple(x)))
                    r = self.mean_explore
                    r[self.sigma_expl > 0] = np.random.normal(r[self.sigma_expl > 0], self.sigma_expl[self.sigma_expl > 0])
                    res = bounds_min_max(r, self.m_mins, self.m_maxs)
                    return res
                else:  # exploit'
                    return array(self.model.infer_order(tuple(x)))

        elif out_dims == self.m_dims[len(self.m_dims)//2:]:  # dm = i(M, S, dS)
            if not self.bootstrapped_s:
                # If only one distinct point has been observed in the sensory space, then we output a random motor command
                return rand_bounds(np.array([self.m_mins[self.m_ndims//2:], self.m_maxs[self.m_ndims//2:]]))[0]
            else:
                assert len(x) == len(in_dims)
                m = x[:self.m_ndims//2]
                s = x[self.m_ndims//2:][:self.s_ndims//2]
                ds = x[self.m_ndims//2:][self.s_ndims//2:]
                self.mean_explore = array(self.model.inv_model.infer_dm(m, s, ds))
                if self.mode == 'explore':
                    r = np.random.normal(self.mean_explore, self.sigma_expl[out_dims])
                    res = bounds_min_max(r, self.m_mins[out_dims], self.m_maxs[out_dims])
                    return res
                else:
                    return self.mean_explore
        else:
            print((in_dims, out_dims, self.m_dims, self.s_dims, self.m_dims[len(self.m_dims)//2:]))
            raise NotImplementedError

    def predict_given_context(self, x, c, c_dims):
        return self.model.inv_model.fwd_model.predict_given_context(x, c, c_dims)

    def update(self, m, s):
        self.model.add_xy(tuple(m), tuple(s))
        self.n_updates += 1
        if not self.bootstrapped_s and self.n_updates > 1:  # make sure there have been at least two updates
            # Make sure we've seen new sensory data since 2 updates ago
            if not list(s) == list(self.model.inv_model.fwd_model.dataset.get_y(self.n_updates - 2)):
                self.bootstrapped_s = True

    def update_batch(self, m_list, s_list):
        self.model.add_xy_batch(m_list, s_list)
        self.n_updates += len(m_list)
        self.bootstrapped_s = True

    def size(self):
        return self.n_updates


sensorimotor_models = {
    'nearest_neighbor': (NonParametric, {'default': {'fwd': 'NN', 'inv': 'NN', 'sigma_explo_ratio':0.1}, # simple nearest neighbor
                                         'exact': {'fwd': 'NN', 'inv': 'NN', 'sigma_explo_ratio':0.},
                                         'dev': {'fwd': 'NN', 'inv': 'NN', 'sigma_explo_ratio':0., 'k': 10, 'learner_kwargs': {'k': 10}}}),
    'WNN': (NonParametric, {'default': {'fwd': 'WNN', 'inv': 'WNN', 'k':20, 'sigma':0.1}}), # weighted nearest neighbor,
    'NSNN': (NonParametric, {'default': {'fwd': 'NSNN', 'inv': 'NSNN', 'sigma_explo_ratio': 0.1},
                             'dev': {'fwd': 'NSNN', 'inv': 'NSNN', 'sigma_explo_ratio': 0.1, 'k': 2, 'learner_kwargs': {'k': 2}}}),
    # LWLR == Locally Weigthed Linear Regression (LWLR)
    # BFGS -> https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
    # 'LWLR-BFGS': (NonParametric, {'default': {'fwd': 'LWLR', 'k':10, 'sigma':0.1, 'inv': 'L-BFGS-B', 'maxfun':50}}),
    # CMAES -> Covariance Matrix Adaptation Evolution Strategy
    'LWLR-NONE': (NonParametric, {'default': {'fwd': 'LWLR', 'k':8, 'sigma':0.1, 'inv': 'NONE'}}),
    'NSLWLR-NONE': (NonParametric, {'default': {'fwd': 'LWLR', 'k':8, 'sigma':0.1, 'inv': 'NONE'}}),
    'NSLWLR-CMAES': (NonParametric, {'default': {'fwd': 'NSLWLR', 'k':10, 'sigma':0.1, 'inv': 'CMAES', 'cmaes_sigma':0.05, 'maxfevals':20}}),

}
