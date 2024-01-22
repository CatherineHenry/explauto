from cozmo.util import Pose, degrees, Angle, speed_mmps, distance_mm

from ..environment import Environment

from ...utils import bounds_min_max

from cozmo.robot import Robot


class CozmoEnvironment(Environment):
    """ 
        Explauto environment for Cozmo robots.

        It can be used with real or simulated robots (using V-REP for instance).
    """
    def __init__(self,
                 cozmo_robot: Robot,
                 move_duration,
                 tracker,
                 m_mins, m_maxs,
                 s_mins, s_maxs,
                 motors=None, # IDK if we need/want with cozmo
                 tracked_obj=None # This seems to be simulation specific (to track named object as the sensori effect?)
                 ):
        """"
        :param cozmo_robot: CozmoCreature instance (it can be a real or a simulated robot)
        :param list motors: list of motors used - it can directly be a motor alias, e.g.m poppy.l_arm # TODO: change to configurable cozmo motor vals. tread movement to start.
        :param float move_duration: duration of the motor commands
        :param tracker: Tracker used to determine the tracked_obj position - when using a robot simulated with V-REP the robot itself can be the tracker.
        :param str tracked_obj: name of the object to track
        :param numpy.array m_mins: minimum motor dims
        :param numpy.array m_maxs: maximum motor dims
        :param numpy.array s_mins: minimum sensor dims
        :param numpy.array s_maxs: maximum sensor dims

        """
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)
        self.robot = cozmo_robot
        self.motors = motors
        self.move_duration = move_duration

        self.tracker = tracker
        self.tracked_obj = tracked_obj

    def compute_motor_command(self, m_ag):
        """ Compute the motor command by restricting it to the bounds. """
        m_env = bounds_min_max(m_ag, self.conf.m_mins, self.conf.m_maxs)
        return m_env

    def compute_sensori_effect(self, m_env):
        """ Move the robot motors and retrieve the tracked object position. """
        # pos = {m.name: pos for m, pos in zip(self.motors, m_env)} # TODO: support motor dicts for head, tread, etc
        # Only support wheels motor for now
        # TODO: can we get idea of "to position" or just speed/acc and duration? maybe Pose?
        # self.robot.drive_wheels(25, 50, self.move_duration)
        # TODO: this doesn't have a move duration. could do a duration wait + stop action if we wanted
        # self.robot.go_to_pose(Pose(m_env[1], 0, 0, angle_z=degrees(m_env[0])), relative_to_robot=False).wait_for_completed()
        # self.robot.turn_in_place(degrees(m_env[0])).wait_for_completed()
        self.robot.turn_in_place(degrees(m_env[0]), angle_tolerance=degrees(0), is_absolute=False, speed=Angle(2)).wait_for_completed()
        self.robot.drive_straight(distance_mm(m_env[1]), speed_mmps(105), should_play_anim=False).wait_for_completed()

        # self.robot.turn_in_place(degrees(m_env[0]), angle_tolerance=degrees(0), is_absolute=True).wait_for_completed()
        # self.robot.drive_straight(distance_mm(m_env[1]), speed_mmps(40), should_play_anim=False).wait_for_completed()

    # self.robot.turn_in_place(angle=degrees(m_env[0]), speed=Angle(m_env[1])).wait_for_completed()
        # This allows to actually apply a motor command
        # Without having a tracker
        if self.tracker is not None: # we want to track camera, not cozmo ( in this case) so return the vector from the camera?
            # return self.tracker.get_object_position(self.tracked_obj)
            return self.tracker[0][0]

    def reset(self):
        """ Resets simulation and does nothing when using a real robot. """
        # if self.robot.simulated:
        #     self.robot.reset_simulation()
        pass  # Cozmo simulation isn't ready, and cozmo obj won't have any concept of simulated

