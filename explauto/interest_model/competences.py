import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Euclidean distance
def competence_dist(target, reached, dist_min=0., dist_max=1.): # target and reached are the (s,m) data so includ
    return max(- dist_max, min(- dist_min, - np.linalg.norm(target - reached)))

def competence_exp(target, reached, dist_min=0., dist_max=1., power=1.):
    comp_dist = competence_dist(target, reached, dist_min, dist_max)
    print(f"Distance between target and reached is {comp_dist}")
    return np.exp(power * comp_dist)


def prediction_error_cos_dist_exp(target, reached):
    # https://www.desmos.com/calculator/ljgnhhfsbk
    # (1-e^-x) is upper bound at 1, and the value remains 0 at 0, adding a value before the e can shift left (if 0<b<1) and shift right (if 1<b) given (1-be^-x).
    # And the rate at which the value approaches 1 can be increased by adding a multiplier to x

    # TODO: do this normalization  better
    # normalize the angle (possible vals -180 and 180)
    target[0] = (target[0] - -180)/(180 - -180)
    reached[0] = (reached[0] - -180)/(180 - -180)

    # normalize the linear movement (possible vals -80 and 80)
    target[1] = (target[1] - -80)/(80 - -80)
    reached[1] = (reached[1] - -80)/(80 - -80)


    cos_sim = cosine_similarity([target], [reached]).flatten()[0]
    cos_dist = 1 - cos_sim  # expecting values between 0 and 2. 0 being equivalent, 1 being orthogonal, and 2 being opposite
    o = 2 # setting o to 2 forces a cos_dist of 2 to the highest possible error of 1
    bounded_cos = 1 - np.exp(-o * cos_dist)
    return cos_sim, cos_dist, bounded_cos

def competence_cos_dist_exp(target, reached):
    prediction_error = prediction_error_cos_dist_exp(target, reached)
    return 1 - prediction_error[2]


def competence_bool(target, reached):
    return float((target == reached).all())
