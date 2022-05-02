import logging
from abc import ABC, abstractmethod

import numpy as np


STOP_THRESHOLD = 0.02


class BehaviouralState(ABC):
    NAME = None

    def __init__(self, behavioural_planner, state_manager):
        self._bp = behavioural_planner
        self._state_manager = state_manager
        assert self.NAME is not None, "Behavioural State does not have a NAME"

    @abstractmethod
    def handle(self, waypoints, ego_state, closed_loop_speed):
        pass

    # Gets the goal index in the list of waypoints, based on the lookahead and
    # the current ego state. In particular, find the earliest waypoint that has accumulated
    # arc length (including closest_len) that is greater than or equal to self._lookahead.
    def _get_goal_index(self, waypoints, ego_state, closest_len, closest_index):
        """Gets the goal index for the vehicle.

        Set to be the earliest waypoint that has accumulated arc length
        accumulated arc length (including closest_len) that is greater than or
        equal to self._lookahead.

        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]:
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
        returns:
            wp_index: Goal index for the vehicle to reach
                i.e. waypoints[wp_index] gives the goal waypoint
        """
        # Find the farthest point along the path that is within the
        # lookahead distance of the ego vehicle.
        # Take the distance from the ego vehicle to the closest waypoint into
        # consideration.
        arc_length = closest_len
        wp_index = closest_index

        # In this case, reaching the closest waypoint is already far enough for
        # the planner.  No need to check additional waypoints.
        if arc_length > self._bp.get_lookahead():
            return wp_index

        # We are already at the end of the path.
        if wp_index == len(waypoints) - 1:
            return wp_index

        # Otherwise, find our next waypoint.
        while wp_index < len(waypoints) - 1:
            arc_length += np.sqrt((waypoints[wp_index][0] - waypoints[wp_index + 1][0]) ** 2 + (
                        waypoints[wp_index][1] - waypoints[wp_index + 1][1]) ** 2)
            if arc_length > self._bp.get_lookahead(): break
            wp_index += 1

        return wp_index % len(waypoints)


class TrackSpeedState(BehaviouralState):
    """
    In this state, continue tracking the lane by finding the
    goal index in the waypoint list that is within the lookahead
    distance. Then, check to see if the waypoint path intersects
    with any stop lines. If it does, then ensure that the goal
    state enforces the car to be stopped before the stop line.
    You should use the get_closest_index(), get_goal_index(), and
    check_for_stop_signs() helper functions.
    Make sure that get_closest_index() and get_goal_index() functions are
    complete, and examine the check_for_stop_signs() function to
    understand it.
    """

    NAME = "TRACK_SPEED"

    def handle(self, waypoints, ego_state, closed_loop_speed):
        logging.debug(f"{self.NAME} STATE (Behavioural Planner)")
        # First, find the closest index to the ego vehicle.
        closest_len, closest_index = get_closest_index(waypoints, ego_state)

        # Next, find the goal index that lies within the lookahead distance
        # along the waypoints.
        goal_index = self._get_goal_index(waypoints, ego_state, closest_len, closest_index)
        while waypoints[goal_index][2] <= 0.1:
            goal_index += 1

        self._bp.set_goal_index(goal_index)
        self._bp.set_goal_state(waypoints[goal_index])


class FollowLeadState(BehaviouralState):
    """
    In this state, we check for lead vehicle within the proximity of the
    ego car, such that the ego car should begin to follow the lead vehicle.
    """

    NAME = "FOLLOW_LEAD"

    def handle(self, waypoints, ego_state, closed_loop_speed):
        pass


class DecelerateToPointState(BehaviouralState):
    """
    In this state, check if we have reached a complete stop. Use the
    closed loop speed to do so, to ensure we are actually at a complete
    stop, and compare to STOP_THRESHOLD.  If so, transition to the next
    state.
    """

    NAME = "DECELERATE_TO_POINT"

    def handle(self, waypoints, ego_state, closed_loop_speed):
        if abs(closed_loop_speed) <= STOP_THRESHOLD:
            self._state_manager.state_transition(StopState.NAME)


class StopState(BehaviouralState):
    """
    In this state, wait until semaphore turns green and there aren't
    any pedestrian. If so, we can now leave the intersection and transition
    to the next state.
    """

    NAME = "STOP"

    def handle(self, waypoints, ego_state, closed_loop_speed):
        # We have stayed stopped for the required number of cycles.
        # Allow the ego vehicle to leave the stop sign. Once it has
        # passed the stop sign, return to lane following.
        # You should use the get_closest_index(), get_goal_index(), and
        # check_for_stop_signs() helper functions.
        closest_len, closest_index = get_closest_index(waypoints, ego_state)
        goal_index = self._get_goal_index(waypoints, ego_state, closest_len, closest_index)
        while waypoints[goal_index][2] <= 0.1:
            goal_index += 1

        # We've stopped for the required amount of time, so the new goal
        # index for the stop line is not relevant. Use the goal index
        # that is the lookahead distance away.
        self._bp.set_goal_index(goal_index)
        self._bp.set_goal_state(waypoints[goal_index])

        # If the stop sign is no longer along our path, we can now
        # transition back to our lane following state.

        # if not stop_sign_found: self._state = FOLLOW_LANE
        self._state_manager.state_transition(TrackSpeedState.NAME)


class EmergencyState(BehaviouralState):
    """
    In this state, we execute an emergency break (uncomfortable break)
    because we detected a pedestrian. We can leave this state after
    EMERGENCY_TIMEOUT is elapsed from when the car has stopped.
    """

    NAME = "EMERGENCY"

    def handle(self, waypoints, ego_state, closed_loop_speed):
        pass


class ShutdownState(BehaviouralState):
    """
    In this state, we don't execute any action due to fatal event.
    """

    NAME = "SHUTDOWN"

    def handle(self, waypoints, ego_state, closed_loop_speed):
        pass


class StateManager:
    INIT_STATE_NAME = StopState.NAME

    def __init__(self, behavioral_planner):
        states = [
            TrackSpeedState(behavioral_planner, self),
            FollowLeadState(behavioral_planner, self),
            StopState(behavioral_planner, self),
            DecelerateToPointState(behavioral_planner, self),
            EmergencyState(behavioral_planner, self),
            ShutdownState(behavioral_planner, self),
        ]
        self._states = {s.NAME: s for s in states}
        self._state = self._states[self.INIT_STATE_NAME]

    def execute(self, waypoints, ego_state, closed_loop_speed):
        logging.debug(f"{self._state.NAME} STATE (Behavioural Planner)")
        self._state.handle(waypoints, ego_state, closed_loop_speed)

    def state_transition(self, new_state_name):
        self._state = self._states[new_state_name]


# Compute the waypoint index that is closest to the ego vehicle, and return
# it as well as the distance from the ego vehicle to that waypoint.
def get_closest_index(waypoints, ego_state):
    """Gets closest index a given list of waypoints to the vehicle position.

    args:
        waypoints: current waypoints to track. (global frame)
            length and speed in m and m/s.
            (includes speed to track at each x,y location.)
            format: [[x0, y0, v0],
                     [x1, y1, v1],
                     ...
                     [xn, yn, vn]]
            example:
                waypoints[2][1]:
                returns the 3rd waypoint's y position

                waypoints[5]:
                returns [x5, y5, v5] (6th waypoint)
        ego_state: ego state vector for the vehicle. (global frame)
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)

    returns:
        [closest_len, closest_index]:
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
    """
    closest_len = float('Inf')
    closest_index = 0

    for i in range(len(waypoints)):
        temp = (waypoints[i][0] - ego_state[0])**2 + (waypoints[i][1] - ego_state[1])**2
        if temp < closest_len:
            closest_len = temp
            closest_index = i
    closest_len = np.sqrt(closest_len)

    return closest_len, closest_index
