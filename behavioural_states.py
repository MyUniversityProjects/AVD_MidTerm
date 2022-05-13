import logging
import math
from abc import ABC, abstractmethod
from datetime import timedelta, datetime

import numpy as np

import constants
from custom_agents import TrafficLightAdapter
from helpers import ego_trajectory_intersect, check_for_path_intersection, angle_between, \
    optimized_dist, ego_pedestrian_intersect

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class BehaviouralState(ABC):
    NAME = None

    def __init__(self, behavioural_planner, state_manager):
        self._bp = behavioural_planner
        self._state_manager = state_manager
        assert self.NAME is not None, "Behavioural State does not have a NAME"

    @abstractmethod
    def handle(self, waypoints, ego_state, closed_loop_speed, pedestrians, traffic_lights):
        pass

    def check_for_traffic_lights(self, waypoints, ego_state, closed_loop_speed, closest_index, goal_index, traffic_lights):
        tl = [i for i in traffic_lights if i.state != TrafficLightAdapter.GREEN]
        agents = [i.get_segment() for i in tl]
        new_goal_index, index = check_for_path_intersection(waypoints, closest_index, goal_index, agents)
        found = index >= 0
        if found:
            traffic_light = tl[index]
            p1, p2 = np.array(ego_state[:2]), np.array(traffic_light.position[:2])
            ori = np.array([math.cos(ego_state[2]), math.sin(ego_state[2])])
            direction_vec = p2 - p1
            angle = angle_between(ori, direction_vec)

            d = math.sqrt(optimized_dist(p1, p2)) * math.cos(angle)
            if d < closed_loop_speed / constants.TRAFFIC_LIGHT_STOP_CONSTANT:
                return goal_index, False
            self._bp.set_traffic_light_id(traffic_light.id)
        return new_goal_index, found

    @staticmethod
    def _check_dangerous_pedestrians(ego_state, waypoints, closest_idx, pedestrians, lookahead=13):
        for ped in pedestrians:
            dangerous = False
            if ego_pedestrian_intersect(ego_state, ped[0], waypoints, closest_idx, ego_lookahead=lookahead):
                logging.info("Dangerous pedestrian found ahead of the vehicle!")
                dangerous = True
            elif ego_trajectory_intersect(ego_state, ped[0], waypoints, closest_idx, ego_lookahead=15, ped_lookahead=5):
                logging.info("Dangerous pedestrian found with trajectory intersection!")
                dangerous = True

            if dangerous:
                return math.sqrt(ped[1])
        return None

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

    @staticmethod
    def _is_green(tl_id, traffic_lights):
        for tl in traffic_lights:
            if tl.id == tl_id:
                return tl.state == TrafficLightAdapter.GREEN
        return False


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

    def handle(self, waypoints, ego_state, closed_loop_speed, pedestrians, traffic_lights):
        # Second, find the closest index to the ego vehicle.
        closest_len, closest_index = get_closest_index(waypoints, ego_state)

        if self._check_dangerous_pedestrians(ego_state, waypoints, closest_index, pedestrians):
            self._state_manager.state_transition(EmergencyState.NAME)
            return

        # Next, find the goal index that lies within the lookahead distance
        # along the waypoints.
        goal_index = self._get_goal_index(waypoints, ego_state, closest_len, closest_index)
        while waypoints[goal_index][2] <= 0.1:
            goal_index += 1

        goal_index, traffic_light_found = self.check_for_traffic_lights(waypoints, ego_state, closed_loop_speed,
                                                                        closest_index, goal_index, traffic_lights)
        goal_state = waypoints[goal_index]

        if traffic_light_found:
            logging.info("TrafficLight found!")
            self._state_manager.state_transition(DecelerateToPointState.NAME)
            goal_state = goal_state[:]
            goal_state[2] = 0

        self._bp.set_goal_index(goal_index)
        self._bp.set_goal_state(goal_state)


class DecelerateToPointState(BehaviouralState):
    """
    In this state, check if we have reached a complete stop. Use the
    closed loop speed to do so, to ensure we are actually at a complete
    stop, and compare to STOP_THRESHOLD.  If so, transition to the next
    state.
    """

    NAME = "DECELERATE_TO_POINT"

    def handle(self, waypoints, ego_state, closed_loop_speed, pedestrians, traffic_lights):
        # Second, find the closest index to the ego vehicle.
        closest_len, closest_index = get_closest_index(waypoints, ego_state)
        tl_id = self._bp.get_traffic_light_id()

        if self._check_dangerous_pedestrians(ego_state, waypoints, closest_index, pedestrians):
            self._state_manager.state_transition(EmergencyState.NAME)
        elif abs(closed_loop_speed) <= constants.STOP_THRESHOLD:
            self._state_manager.state_transition(StopState.NAME)
        elif tl_id is None or self._is_green(tl_id, traffic_lights):
            self._bp.set_traffic_light_id(None)
            self._state_manager.state_transition(TrackSpeedState.NAME)


class StopState(BehaviouralState):
    """
    In this state, wait until traffic light turns green and there aren't
    any pedestrian. If so, we can now leave the intersection and transition
    to the next state.
    """

    NAME = "STOP"

    def handle(self, waypoints, ego_state, closed_loop_speed, pedestrians, traffic_lights):
        # We have stayed stopped until traffic light turns green and there
        # aren't any pedestrian.
        tl_id = self._bp.get_traffic_light_id()
        if tl_id is None or self._is_green(tl_id, traffic_lights):
            _, closest_index = get_closest_index(waypoints, ego_state)
            if self._check_dangerous_pedestrians(ego_state, waypoints, closest_index, pedestrians):
                self._bp.set_traffic_light_id(None)
                self._state_manager.state_transition(TrackSpeedState.NAME)


class EmergencyState(BehaviouralState):
    """
    In this state, we execute an emergency break (uncomfortable break)
    because we detected a pedestrian. We can leave this state after
    EMERGENCY_TIMEOUT is elapsed from when the car has stopped.
    """

    NAME = "EMERGENCY"
    TIMEOUT = timedelta(seconds=2)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_time = datetime.now()
        self._last_is_stopped = False

    def handle(self, waypoints, ego_state, closed_loop_speed, pedestrians, traffic_lights):
        speed = abs(closed_loop_speed)
        curr_is_stopped = speed <= constants.STOP_THRESHOLD

        if not curr_is_stopped:
            _, closest_index = get_closest_index(waypoints, ego_state)
            distance = self._check_dangerous_pedestrians(ego_state, waypoints, closest_index, pedestrians)
            if distance is not None:
                brake_value = self._compute_brake_value(speed, distance)
                self._bp.set_emergency_brake_value(brake_value)

        if curr_is_stopped and not self._last_is_stopped:
            logging.info('Starting emergency timer')
            self._last_is_stopped = True
            self._start_time = datetime.now()
        elif self._last_is_stopped and datetime.now() >= self._start_time + self.TIMEOUT:
            logging.info('Emergency timer elapsed')
            self._start_time = datetime.now()
            self._last_is_stopped = False
            self._state_manager.state_transition(StopState.NAME)

    @staticmethod
    def _compute_brake_value(speed, distance):
        ratio = constants.BRAKE_PARAM * speed / distance - 0.2
        ratio = max(min(ratio, 1), 0)
        logging.info(f'SPEED={speed:.2f} DISTANCE={distance:.2f} => EMERGENCY_BRAKE={ratio:.2f}')
        return ratio


class StateManager:
    INIT_STATE_NAME = StopState.NAME

    def __init__(self, behavioral_planner):
        states = [
            TrackSpeedState(behavioral_planner, self),
            StopState(behavioral_planner, self),
            DecelerateToPointState(behavioral_planner, self),
            EmergencyState(behavioral_planner, self),
        ]
        self._states = {s.NAME: s for s in states}
        self._state = self._states[self.INIT_STATE_NAME]

    def get_state(self):
        return self._state

    def execute(self, waypoints, ego_state, closed_loop_speed, pedestrians, traffic_lights):
        self._state.handle(waypoints, ego_state, closed_loop_speed, pedestrians, traffic_lights)

    def state_transition(self, new_state_name):
        logging.info(f"StateChange: {self._state.NAME} => {new_state_name} (Behavioural Planner)")
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
