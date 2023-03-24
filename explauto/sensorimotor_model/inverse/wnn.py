
import numpy as np

from explauto.utils import dist, gaussian_kernel
from .. import forward
from . import inverse


class WeightedNNInverseModel(inverse.InverseModel):
    """Weighted Nearest Neighbors Inverse Model"""

    name = 'WNN'
    desc = 'Weighted Nearest Neighbors'

    def __init__(self, dim_x, dim_y, fwd_model, sigma=1.0, k=10, **kwargs):
        """
        @param k      the number of neighbors to consider for averaging
        @param sigma  for the moment, default sigma for forward model is the same as
                      the one of the inverse model. Not ideal. #FIXME
        """
        self.k = fwd_model.k
        self.fwd_model = fwd_model
        self.sigma  = sigma

    def infer_x(self, y, sigma=None, k=None, **kwargs):
        """Infer probable x from input y

        @param y  the desired output for infered x.
        @param k  how many neighbors to consider for the average
                  this value override the class provided one on a per
                  method call basis.
        """
        assert len(y) == self.fwd_model.dim_y, "Wrong dimension for y. Expected %i, got %i" % (self.fwd_model.dim_y, len(y))
        k = k or self.k
        sigma = sigma or self.sigma
        x_guess = self._guess_x(y, k = k)[0]
        distances, index = self.fwd_model.dataset.nn_x(x_guess, k = k)
        w = self._weights(index, distances, sigma*sigma,  y)
        return [np.sum([wi*self.fwd_model.dataset.get_x(idx)
                for wi, idx in zip(w, index)], axis = 0)]

    def _weights(self, index, distances, sigma_sq, y_desired):

        distances = [dist(self.fwd_model.dataset.get_y(idx), y_desired)
                     for idx in index] # could be optimized

        w = np.fromiter((gaussian_kernel(d / self.fwd_model.dim_y, sigma_sq)
                         for d in distances), np.float)

        # We eliminate the outliers # TODO : actually reduce w and index
        wsum = w.sum()
        if wsum == 0:
            return 1.0 / len(distances) * np.ones((len(distances),))
        else:
            eps = wsum * 1e-15 / self.fwd_model.dim_y
            return np.fromiter((w_i/wsum if w_i > eps else 0.0 for w_i in w), np.float)


class ESWNNInverseModel(WeightedNNInverseModel):
    """ES-WNN : WNN with estimated sigma, on a query basis, as the mean distance."""

    name = 'ES-WNN'

    def _weights(self, index, distances, sigma_sq, y_desired):
        sigma_sq= (distances ** 2).sum() / len(distances)
        return WeightedNNInverseModel._weights(self, index, distances, sigma_sq, y_desired)
