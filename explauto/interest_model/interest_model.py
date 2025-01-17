from abc import ABCMeta, abstractmethod

from . import interest_models

class InterestModel(object, metaclass=ABCMeta):
    def __init__(self, expl_dims):
        self.expl_dims = expl_dims

    # take kwargs so that any kw args not in the config (so, non static) can be passed
    @classmethod
    def from_configuration(cls, conf, expl_dims, im_name, config_name='default', **kwargs):
        im_cls, im_configs = interest_models[im_name]
        return im_cls(conf, expl_dims, **im_configs[config_name], **kwargs)

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def update(self, xy, ms):
        pass
