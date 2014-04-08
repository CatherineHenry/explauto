from .. import Environment
from numpy import pi, array, ones
from ...utils.utils import bounds_min_max
from ...agent.motor_primitive import BasisFunctions
import simple_lip 
from numpy.random import randn


class PendulumEnvironment(Environment):
    #def __init__(self, m_ndims, s_ndims, m_mins, m_maxs, length_ratio, noise):
    def __init__(self, **kwargs):
        for attr in ['m_mins', 'm_maxs', 's_mins', 's_maxs', 'noise']:
            setattr(self, attr, kwargs[attr])
        self.m_ndims =len(self.m_mins) 
        self.s_ndims =len(self.s_mins) 
        Environment.__init__(self, ndims=self.m_ndims + self.s_ndims)
        #self.m_mins = m_mins
        #self.m_maxs = m_maxs
        #self.noise = noise
        self.readable = range(self.m_ndims + self.s_ndims)
        self.writable =  range(self.m_ndims)
        self.x0 = [-pi, 0.]
        self.dt = 0.25
        sekf.bf = BasisFunctions(self.m_ndims, 150*self.dt, self.dt, 4.)


    def next_state(self, ag_state):
        m = ag_state
        self.state[:self.m_ndims] = bounds_min_max(m, self.m_mins, self.m_maxs)
        s = self.x0
        for u in bf.trajectory(self.state[:self.m_ndims]).flatten():
            s = simple_lip.simulate(s, [u], self.dt)
        res = array(s)
        res += self.noise * randn(*res.shape)
        self.state[-self.s_ndims:] = res
        

