import math
from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt


class PlotObject:
    def __init__(self, image_path, nose_adj_dist, nose_opp_dist, tail_adj_dist, tail_opp_dist, leftmost="nose"):
        '''
        # :param object_label a way of distinguishing this physical object from another. not a label used to associate meaning to the concept
        :param image_path the path to the image
        :param nose_adj_dist the adjacent distance (along 0 axis) from cozmo to the object nose
        :param nose_opp_dist the opposite distance (from the 0 axis) to the object nose
        :param tail_adj_dist the adjacent distance to the obj tail
        :param tail_opp_dist the opposite distance to the obj tail

        The "leftness" and "rightness" are w.r.t left and right when cozmo is facing the object
        '''

        # self.object_label = object_label
        self.image_path = image_path
        self.nose_adj_dist = nose_adj_dist
        self.nose_opp_dist = nose_opp_dist
        self.tail_adj_dist = tail_adj_dist
        self.tail_opp_dist = tail_opp_dist
        self.leftmost = leftmost
        self.nose_angle_from_0 = self.get_nose_angle_from_0()
        self.tail_angle_from_0 = self.get_tail_angle_from_0()
        self.leftmost_angle_from_0 = self.nose_angle_from_0 if leftmost == "nose" else self.tail_angle_from_0
        self.rightmost_angle_from_0 = self.tail_angle_from_0 if leftmost == "nose" else self.nose_angle_from_0
        self.angle_from_0_avg = (self.leftmost_angle_from_0 + self.rightmost_angle_from_0)/2
        self.image = np.fliplr(plt.imread(self.image_path)) if leftmost == "tail" else plt.imread(self.image_path)
        self.flip_image = True if leftmost == "tail" else False  # this works because all photos should be nose to the left, for consistency.

    def get_nose_angle_from_0(self):
        nose_angle_from_0 = math.degrees(math.atan2(self.nose_opp_dist, self.nose_adj_dist))
        # if radian: dont need after all
        #     nose_angle_from_0 = nose_angle_from_0 + 360 if nose_angle_from_0 < 0 else nose_angle_from_0  # add 360 so our degree spans 0 to 360
        return nose_angle_from_0

    def get_tail_angle_from_0(self):
        tail_angle_from_0 = math.degrees(math.atan2(self.tail_opp_dist, self.tail_adj_dist))
        # if radian: dont need after all
        #     tail_angle_from_0 = tail_angle_from_0 + 360 if tail_angle_from_0 < 0 else tail_angle_from_0  # add 360 so our degree spans 0 to 360
        return tail_angle_from_0




