
import numpy as np

from .inverse import InverseModel


class UnusedInverseModel(InverseModel):
    """An unused/placeholder Inverse Model."""


    name = "Unused"
    desc = 'Unused'


    @classmethod
    def from_dataset(cls, dataset, sigma, **kwargs):
        pass


    def __init__(self, dim_x, dim_y, fwd_model, **kwargs):
        """
        @param k  the number of neighbors to consider for averaging
        """
        # self.dim_x = dim_x
        # self.dim_y = dim_y
        InverseModel.__init__(self, dim_x, dim_y, **kwargs)
        self.fwd_model = fwd_model
        self.k = fwd_model.k
