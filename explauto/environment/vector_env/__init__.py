from .vector_env import  VectorEnvironment

# Cannot mock up a configuration the same way as simple_arm, for example, because computing the sensori is done by the robot moving + camera input.
# TODO: revisit when we have a simulation in place?
