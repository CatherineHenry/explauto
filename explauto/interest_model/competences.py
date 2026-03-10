import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Euclidean distance (distance from origin)
def competence_dist(target, reached, dist_min=0., dist_max=1.): # target and reached are the (s,m) data so includ
    return max(- dist_max, min(- dist_min, - np.linalg.norm(target - reached)))  # this norm essentially is euclidean distance


def competence_exp(target, reached, dist_min=0., dist_max=1., power=1.):
    comp_dist = competence_dist(target, reached, dist_min, dist_max)
    print(f"Distance between target and reached is {comp_dist}")
    return np.exp(power * comp_dist)


def prediction_error_cos_dist_exp(target, reached, bounds):
    # https://www.desmos.com/calculator/ljgnhhfsbk
    # (1-e^-x) is upper bound at 1, and the value remains 0 at 0, adding a value before the e can shift left (if 0<b<1) and shift right (if 1<b) given (1-be^-x).
    # And the rate at which the value approaches 1 can be increased by adding a multiplier to x

    # TODO: do this normalization  better
    # normalize the angle (possible vals -180 and 180)
    bounds_mins = bounds[0]
    bounds_maxs = bounds[1]
    # normalize movement along Z (rotation) (should be -180 to 180)
    target[2] = (target[2] - bounds_mins[2])/(bounds_maxs[2] - bounds_mins[2])
    reached[2] = (reached[2] - bounds_mins[2])/(bounds_maxs[2] - bounds_mins[2])

    # normalize the linear movement along X (for example, possible vals could look like -80 and 80)
    target[0] = (target[0] - bounds_mins[0])/(bounds_maxs[0] - bounds_mins[0])
    reached[0] = (reached[0] - bounds_mins[0])/(bounds_maxs[0] - bounds_mins[0])

    # normalize the linear movement along Y (for example, possible vals could look like -80 and 80)
    target[1] = (target[1] - bounds_mins[1])/(bounds_maxs[1] - bounds_mins[1])
    reached[1] = (reached[1] - bounds_mins[1])/(bounds_maxs[1] - bounds_mins[1])

    # expecting values between -1 and 1. 1 being equivalent, 0 being orthogonal, and -1 being opposite
    cos_sim = cosine_similarity([target], [reached]).flatten()[0]
    # expecting values between 0 and 2. 0 being equivalent, 1 being orthogonal, and 2 being opposite
    cos_dist = 1 - cos_sim
    o = 2 # setting o to 2 forces a cos_dist of 2 to the highest possible error of 1
    bounded_cos = 1 - np.exp(-o * cos_dist)
    return cos_sim, cos_dist, bounded_cos


def competence_cos_dist_exp(target, reached, bounds):
    prediction_error = prediction_error_cos_dist_exp(target, reached, bounds)
    return 1 - prediction_error[2]


def competence_bool(target, reached):
    return float((target == reached).all())
