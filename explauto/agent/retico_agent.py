import logging
from datetime import datetime

import numpy as np


from ..utils.config import make_configuration
from ..utils.observer import Observable
from ..utils import rand_bounds, bounds_min_max
from ..exceptions import ExplautoBootstrapError

logger = logging.getLogger(__name__)


class ReticoAgent(Observable):
    def __init__(self, conf, sm_model, im_model, n_bootstrap=0, execution_uuid = None, execution_date_timestamp = None, context_mode=None, save_data=False, experiment_name=None):
        Observable.__init__(self)
        self.conf = conf
        self.ms = np.zeros(self.conf.ndims)
        self.expl_dims = im_model.expl_dims
        self.inf_dims = sorted(list(set(conf.dims) - set(self.expl_dims)))

        self.sensorimotor_model = sm_model
        self.interest_model = im_model

        # self.competence = competence
        self.n_perceived = 0
        self.n_bootstrap = n_bootstrap
        self.context_mode = context_mode
        self.execution_uuid = execution_uuid
        self.execution_date_timestamp = execution_date_timestamp
        self.save_data = save_data
        self.experiment_name = experiment_name


    @classmethod
    def from_classes(cls,
                 im_model_cls, im_model_config, expl_dims,
                 sm_model_cls, sm_model_config, inf_dims,
                 m_mins, m_maxs, s_mins, s_maxs, n_bootstrap=0, context_mode=None):
        """Initialize agent class

        :param class im_model_cls: a subclass of InterestedModel, as those registered in the interest_model package

        :param dict im_model_config: a configuration dict as those registered in the interest_model package

        :param list expl_dims: the sensorimotor dimensions where exploration is driven in the interest model

        :param list inf_dims: the output sensorimotor dimensions of the sensorimotor model (input being expl_dims)

        :param class sm_model_cls: a subclass of SensorimotorModel, as those registered in the sensorimotor_model package

        :param dict sensorimotor_model_config: a configuration dict as those registered in the sensorimotor_model package

        :param list m_mins, m_maxs, s_mins, s_max: lower and upper bounds of motor and sensory values on each dimension

        """

        conf = make_configuration(m_mins, m_maxs, s_mins, s_maxs)

        sm_model = sm_model_cls(conf, **sm_model_config)
        im_model = im_model_cls(conf, expl_dims,
                                           **im_model_config)

        return cls(conf, sm_model, im_model, n_bootstrap, context_mode)


    def choose(self, context_ms=None):
        """ Returns a point chosen by the interest model
        """
        try:
            if self.context_mode is None:
                x = self.interest_model.sample()
            else:
                if self.context_mode["mode"] == 'mdmsds':
                    if self.expl_dims == self.conf.s_dims:
                        x = np.hstack((context_ms[self.conf.m_ndims//2:], self.interest_model.sample_given_context(context_ms[self.conf.m_ndims//2:], list(range(self.conf.s_ndims//2)))))
                    else:
                        if self.context_mode['choose_m']: # 'choose_m' parameter defines if the robot is allowed to choose $m$ and $\\Delta m$ at each iteration instead of only $\\Delta m$.
                            x = self.interest_model.sample()
                        else:
                            x = np.hstack((context_ms[:self.conf.m_ndims//2], self.interest_model.sample_given_context(context_ms[:self.conf.m_ndims//2], list(range(self.conf.m_ndims//2)))))                
                elif self.context_mode["mode"] == 'mcs': # The context is hidden in self.current_context, and not available to the learning agent. We thus use the \"ContextEnvironment\" class to translate this HalfLazyArm environment to an environment which is providing the context in the sensory feedback. The sensory feedback is now the concatenation of the context $c$ and the feedback of the arm $s$.\n", "We thus call this mode \"mcs\
                    x = np.hstack((context_ms, self.interest_model.sample_given_context(context_ms, list(range(self.context_mode["context_n_dims"])))))
        except ExplautoBootstrapError:
            logger.warning('Interest model not bootstrapped yet')
            x = rand_bounds(self.conf.bounds[:, self.expl_dims]).flatten()
            if self.context_mode is not None:
                x = x[list(set(self.expl_dims) - set(self.context_mode['context_dims']))]
        return x

    def infer(self, expl_dims, inf_dims, x):
        """ Use the sensorimotor model to compute the expected value on inf_dims given that the value on expl_dims is x.

        .. note:: This corresponds to a prediction if expl_dims=self.conf.m_dims and inf_dims=self.conf.s_dims and to inverse prediction if expl_dims=self.conf.s_dims and inf_dims=self.conf.m_dims.
        """
        try:
            if self.n_bootstrap > 0:
                self.n_bootstrap -= 1
                raise ExplautoBootstrapError
            y = self.sensorimotor_model.infer(expl_dims,
                                              inf_dims,
                                              x.flatten())
        except ExplautoBootstrapError:
            logger.warning('Sensorimotor model not bootstrapped yet')
            y = rand_bounds(self.conf.bounds[:, inf_dims]).flatten()
        return y

    def extract_ms(self, x, y):
        """ Returns the motor and sensory parts from a point in the exploration
        space (expl_dims) and a point in the inference space (inf_dims).
        """
        ms = np.zeros(self.conf.ndims)
        ms[self.expl_dims] = x
        ms[self.inf_dims] = y
        return ms[self.conf.m_dims], ms[self.conf.s_dims]

    def motor_primitive(self, m):
        """ Prepare the movement from a command m. To be overridded in order to generate more complex movement (tutorial to come). This version simply bounds the command.
        """
        return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)

    def sensory_primitive(self, s):
        """ Extract features from a sensory effect s. To be overridded in order to process more complex feature extraction (tutorial to come). This version simply bounds the sensory effect.
        """
        return bounds_min_max(s, self.conf.s_mins, self.conf.s_maxs)

    def produce(self, context_ms=None, flow_uuid=None):
        """ Exploration (see the `Explauto introduction <about.html>`__ for more detail):

        * Choose a value x on expl_dims according to the interest model
        * Infer a value y on inf_dims from x using the :meth:`~explauto.agent.agent.Agent.infer` method
        * Extract the motor m and sensory s parts of x, y
        * generate a movement from m

        :returns: the generated movement

        .. note:: This correspond to motor babbling if expl_dims=self.conf.m_dims and inf_dims=self.conf.s_dims and to  goal babbling if expl_dims=self.conf.s_dims and inf_dims=self.conf.m_dims.
        """
        if context_ms is None:
            self.x = self.choose() # choose the next motor or sensory goal (depending on exploration space)
            self.y = self.infer(self.expl_dims, self.inf_dims, self.x)
        else:
            if self.context_mode["mode"] == 'mdmsds':
                self.x = self.choose(context_ms) 
                if self.expl_dims == self.conf.s_dims and not self.context_mode['choose_m']:
                    m = context_ms[:self.conf.m_ndims//2]
                    in_dims = list(range(self.conf.m_ndims//2)) + list(range(self.conf.m_ndims, self.conf.m_ndims + self.conf.s_ndims))
                    out_dims = list(range(self.conf.m_ndims//2, self.conf.m_ndims))
                    dm = self.infer(in_dims, 
                                    out_dims, 
                                    np.array(m + list(self.x)))
                    self.y = np.hstack((m, dm))
                else:
                    self.y = self.infer(self.expl_dims, self.inf_dims, self.x)
            elif self.context_mode["mode"] == 'mcs':
                self.x = self.choose(context_ms) 
                self.y = self.infer(self.expl_dims, self.inf_dims, self.x)
                

        # chosen motor goal, predicted sensory effect
        self.m, self.s = self.extract_ms(self.x, self.y)

        movement = self.motor_primitive(self.m)

        self.emit(f'[{flow_uuid}] choice', self.x)
        self.emit(f'[{flow_uuid}] inference', list(self.y)) # change to list so it doesn't wrap. temp solution
        self.emit(f'[{flow_uuid}] movement', movement)

        return movement


    def perceive(self, s_, context=None, flow_uuid=None):
        """ Learning (see the `Explauto introduction <about.html>`__ for more detail):

        * update the sensorimotor model with (m, s)
        * update the interest model with (x, y, m, s)
          (x, y are stored in self.ms in :meth:`~explauto.agent.agent.Agent.production`)
        """
        s = self.sensory_primitive(s_)
        self.emit(f'[{flow_uuid}] perception', list(s))
        if context is None:                
            self.sensorimotor_model.update(self.m, s)
            self.interest_model.update(np.hstack((self.m, self.s)), np.hstack((self.m, s)), flow_uuid=flow_uuid)
            if self.save_data and self.n_perceived > 0 and self.n_perceived % 5 == 0:  # Every 5 perceived
                self.save(f"./IAC_output_data/agent_{self.execution_uuid}.pickle")
        else:
            if self.context_mode["mode"] == 'mdmsds':  
                m = self.m[:len(self.m)//2]
                dm = self.m[len(self.m)//2:]
                ds = s[len(s)//2:]
                s = s[:len(s)//2]
                ds_g = self.s[:len(self.s)//2]
                self.sensorimotor_model.update(np.hstack((m, dm)), np.hstack((s, ds)))
                self.interest_model.update(np.hstack((m, dm, context, ds_g)), np.hstack((m, dm, s, ds)))
            elif self.context_mode["mode"] == 'mcs':
                s = s[len(context):]
                s_g = self.s[len(context):]
                self.sensorimotor_model.update(self.m, np.hstack((context, s)))
                self.interest_model.update(np.hstack((self.m, context, s_g)), np.hstack((self.m, context, s)))
                
        self.n_perceived += 1

    def save(self, filename, mode='pickle'):
        self.emit(f'[{self.execution_uuid}] {datetime.now().strftime("%H:%M:%S")}', f'Saving {filename}')
        if mode == 'pickle':
            import pickle
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise NotImplementedError('{} is not implemented'.format(mode))
