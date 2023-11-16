import math
from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt


class PlotObject:
    def __init__(self, image_path, nose_x, nose_y, tail_x, tail_y, leftmost="nose"):
        '''
        # :param object_label a way of distinguishing this physical object from another. not a label used to associate meaning to the concept
        :param image_path the path to the image
        :param nose_x the adjacent distance (along 0 axis) from cozmo to the object nose
        :param nose_y the opposite distance (from the 0 axis) to the object nose
        :param tail_x the adjacent distance to the obj tail
        :param tail_y the opposite distance to the obj tail

        The "leftness" and "rightness" are w.r.t left and right when cozmo is facing the object
        '''

        # NOTE: The concept of leftmost doesn't work well on objects that are parallel with cozmo

        # self.object_label = object_label
        self.image_path = image_path
        self.nose_x = nose_x # adjacent
        self.nose_y = nose_y # opposite
        self.tail_x = tail_x # adjacent
        self.tail_y = tail_y # opposite

        # calculate NT vector (T point - N point)
        self.NT_vec = (tail_x - nose_x, tail_y - nose_y)
        # this only works because arctan2 takes into account quadrant correctly!
        self.NT_vec_angle = math.degrees(math.atan2(self.NT_vec[1], self.NT_vec[0]))
        # because cozmo turns left for objects in first and second quadrant, will see nose first on an object with nose to cozmo right
        if  nose_y > 0: # First or Second Quadrant
            # this >= -90 only works for objects straddling quadrant 1 and 4. is ok because cozmo doesn't turn full 180 from initial position
            if 90 >= self.NT_vec_angle >= -90:
                self.leftmost = "nose"
            else:
                self.leftmost = "tail"
        # because cozmo turns right for objects in third and fourth quadrant, will see tail first on an object with nose to cozmo right
        elif nose_y <= 0: # Third or Fourth Quadrant
            if 90 > self.NT_vec_angle > -90:
                self.leftmost = "tail"
            else:
                self.leftmost = "nose"

        self.nose_angle_from_0 = self.get_nose_angle_from_0()
        self.tail_angle_from_0 = self.get_tail_angle_from_0()
        self.leftmost_angle_from_0 = self.nose_angle_from_0 if self.leftmost == "nose" else self.tail_angle_from_0
        self.rightmost_angle_from_0 = self.tail_angle_from_0 if self.leftmost == "nose" else self.nose_angle_from_0
        self.angle_from_0_avg = (self.leftmost_angle_from_0 + self.rightmost_angle_from_0)/2
        # if the obj is straddling the negative x axis, we need to adjust
        if self.rightmost_angle_from_0 > 0 and self.leftmost_angle_from_0 < 0: # straddling -x axis
            self.angle_from_0_avg = (abs(self.leftmost_angle_from_0) + self.rightmost_angle_from_0)/2
        else:
            self.angle_from_0_avg = (self.leftmost_angle_from_0 + self.rightmost_angle_from_0)/2


        # the images are flipped backwards for the grid plot because it is the rotation laid flat, weird to think about + difficult to explain atm
        self.grid_image = np.fliplr(plt.imread(self.image_path)) if self.leftmost == "nose" else plt.imread (self.image_path)

        # special flip treatment depending on if obj is in quadrant 1 or 2. Need opposite flip rule when obj is in bottom quadrants
        # because on bottom quadrants cozmo "observes" objects from opposing side of our top down view
        # flipping is only for a more interpretable view in the polar plot. not flipping would be a more accurate representation of
        # cozmos view when moving from 0,0
        if nose_y > 0:
            self.polar_image = np.fliplr(plt.imread(self.image_path)) if self.leftmost == "tail" else plt.imread(self.image_path)
        else:
            self.polar_image = np.fliplr(plt.imread(self.image_path)) if self.leftmost == "nose" else plt.imread(self.image_path)

    def get_nose_angle_from_0(self):
        nose_angle_from_0 = math.degrees(math.atan2(self.nose_y, self.nose_x))
        # if radian: dont need after all
        #     nose_angle_from_0 = nose_angle_from_0 + 360 if nose_angle_from_0 < 0 else nose_angle_from_0  # add 360 so our degree spans 0 to 360
        return nose_angle_from_0

    def get_tail_angle_from_0(self):
        tail_angle_from_0 = math.degrees(math.atan2(self.tail_y, self.tail_x))
        # if radian: dont need after all
        #     tail_angle_from_0 = tail_angle_from_0 + 360 if tail_angle_from_0 < 0 else tail_angle_from_0  # add 360 so our degree spans 0 to 360
        return tail_angle_from_0




