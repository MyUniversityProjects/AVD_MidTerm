import logging
from abc import ABC, abstractmethod

import numpy as np

from custom_agents import TrafficLightAdapter

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


STOP_THRESHOLD = 0.02


class BehaviouralState(ABC):
    NAME = None

    def __init__(self, behavioural_planner, state_manager):
        self._bp = behavioural_planner
        self._state_manager = state_manager
        assert self.NAME is not None, "Behavioural State does not have a NAME"

    @abstractmethod
    def handle(self, waypoints, ego_state, closed_loop_speed, traffic_lights):
        pass

    def _check_for_path_intersection(self, waypoints, closest_index, goal_index, agents):
        """Checks for an agent that is intervening the goal path.

        Checks for an agent that is intervening the goal path. Returns a new
        goal index (the current goal index is obstructed by a stop line), and a
        boolean flag indicating if an agent obstruction was found.

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
                closest_index: index of the waypoint which is closest to the vehicle.
                    i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
                goal_index (current): Current goal index for the vehicle to reach
                    i.e. waypoints[goal_index] gives the goal waypoint
        variables to set:
            [goal_index (updated), agent_found]: 
                goal_index (updated): Updated goal index for the vehicle to reach
                    i.e. waypoints[goal_index] gives the goal waypoint
                agent_found: Boolean flag for whether an agent was found or not
        """
        # TODO: cambiare i nomi delle variabili e farli generali per agent
        for i in range(closest_index, goal_index):
            # Check to see if path segment crosses any of the stop lines.
            intersect_flag = False
            for index, agent in enumerate(agents):

                wp_1 = np.array(waypoints[i][0:2])
                wp_2 = np.array(waypoints[i+1][0:2])
                s_1 = np.array(agent[0:2])
                s_2 = np.array(agent[2:4])

                v1 = np.subtract(wp_2, wp_1)
                v2 = np.subtract(s_1, wp_2)
                sign_1 = np.sign(np.cross(v1, v2))
                v2 = np.subtract(s_2, wp_2)
                sign_2 = np.sign(np.cross(v1, v2))

                v1 = np.subtract(s_2, s_1)
                v2 = np.subtract(wp_1, s_2)
                sign_3 = np.sign(np.cross(v1, v2))
                v2 = np.subtract(wp_2, s_2)
                sign_4 = np.sign(np.cross(v1, v2))

                # Check if the line segments intersect.
                if (sign_1 != sign_2) and (sign_3 != sign_4):
                    intersect_flag = True

                # Check if the collinearity cases hold.
                if (sign_1 == 0) and self._pointOnSegment(wp_1, s_1, wp_2):
                    intersect_flag = True
                if (sign_2 == 0) and self._pointOnSegment(wp_1, s_2, wp_2):
                    intersect_flag = True
                if (sign_3 == 0) and self._pointOnSegment(s_1, wp_1, s_2):
                    intersect_flag = True
                if (sign_3 == 0) and self._pointOnSegment(s_1, wp_2, s_2):
                    intersect_flag = True

                # If there is an intersection with a stop line, update
                # the goal state to stop before the goal line.
                if intersect_flag:
                    goal_index = i
                    return goal_index, index

        return goal_index, -1

    def _pointOnSegment(self, p1, p2, p3):
        if (p2[0] <= max(p1[0], p3[0]) and (p2[0] >= min(p1[0], p3[0])) and
        (p2[1] <= max(p1[1], p3[1])) and (p2[1] >= min(p1[1], p3[1]))):
            return True
        else:
            return False

    def check_for_traffic_lights(self, waypoints, ego_state, closest_index, goal_index, traffic_lights):
        tl = [i for i in traffic_lights if i.state != TrafficLightAdapter.GREEN]
        agents = [i.get_segment() for i in tl]
        goal_index, index = self._check_for_path_intersection(waypoints, closest_index, goal_index, agents)
        found = index >= 0
        if found:
            self._bp.set_traffic_light_id(tl[index].id)
        return goal_index, found

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

    def handle(self, waypoints, ego_state, closed_loop_speed, traffic_lights):
        # First, find the closest index to the ego vehicle.
        closest_len, closest_index = get_closest_index(waypoints, ego_state)

        # Next, find the goal index that lies within the lookahead distance
        # along the waypoints.
        goal_index = self._get_goal_index(waypoints, ego_state, closest_len, closest_index)
        while waypoints[goal_index][2] <= 0.1:
            goal_index += 1

        goal_index, traffic_light_found = self.check_for_traffic_lights(waypoints, ego_state, closest_index, goal_index, traffic_lights)
        goal_state = waypoints[goal_index]

        if traffic_light_found:
            logging.info("Ho trovato un semaforo")
            self._state_manager.state_transition(DecelerateToPointState.NAME)
            goal_state = goal_state[:]
            goal_state[2] = 0

        self._bp.set_goal_index(goal_index)
        self._bp.set_goal_state(goal_state)


class FollowLeadState(BehaviouralState):
    """
    In this state, we check for lead vehicle within the proximity of the
    ego car, such that the ego car should begin to follow the lead vehicle.
    """

    NAME = "FOLLOW_LEAD"

    def handle(self, waypoints, ego_state, closed_loop_speed, traffic_lights):
        pass


class DecelerateToPointState(BehaviouralState):
    """
    In this state, check if we have reached a complete stop. Use the
    closed loop speed to do so, to ensure we are actually at a complete
    stop, and compare to STOP_THRESHOLD.  If so, transition to the next
    state.
    """

    NAME = "DECELERATE_TO_POINT"

    def handle(self, waypoints, ego_state, closed_loop_speed, traffic_lights):
        if abs(closed_loop_speed) <= STOP_THRESHOLD:
            self._state_manager.state_transition(StopState.NAME)


class StopState(BehaviouralState):
    """
    In this state, wait until semaphore turns green and there aren't
    any pedestrian. If so, we can now leave the intersection and transition
    to the next state.
    """

    NAME = "STOP"

    def handle(self, waypoints, ego_state, closed_loop_speed, traffic_lights):
        # We have stayed stopped for the required number of cycles.
        # Allow the ego vehicle to leave the stop sign. Once it has
        # passed the stop sign, return to lane following.
        # You should use the get_closest_index(), get_goal_index(), and
        # check_for_stop_signs() helper functions.

        # closest_len, closest_index = get_closest_index(waypoints, ego_state)
        # goal_index = self._get_goal_index(waypoints, ego_state, closest_len, closest_index)
        # while waypoints[goal_index][2] <= 0.1:
        #     goal_index += 1

        # We've stopped for the required amount of time, so the new goal
        # index for the stop line is not relevant. Use the goal index
        # that is the lookahead distance away.

        # self._bp.set_goal_index(goal_index)
        # self._bp.set_goal_state(waypoints[goal_index])

        # If the stop sign is no longer along our path, we can now
        # transition back to our lane following state.

        # if not stop_sign_found: self._state = FOLLOW_LANE
        tl_id = self._bp.get_traffic_light_id()
        if tl_id is None or self._is_green(tl_id, traffic_lights):
            self._bp.set_traffic_light_id(None)
            self._state_manager.state_transition(TrackSpeedState.NAME)

    def _is_green(self, tl_id, traffic_lights):
        for tl in traffic_lights:
            if tl.id == tl_id:
                return tl.state == TrafficLightAdapter.GREEN
        return False


class EmergencyState(BehaviouralState):
    """
    In this state, we execute an emergency break (uncomfortable break)
    because we detected a pedestrian. We can leave this state after
    EMERGENCY_TIMEOUT is elapsed from when the car has stopped.
    """

    NAME = "EMERGENCY"

    def handle(self, waypoints, ego_state, closed_loop_speed, traffic_lights):
        pass


class ShutdownState(BehaviouralState):
    """
    In this state, we don't execute any action due to fatal event.
    """

    NAME = "SHUTDOWN"

    def handle(self, waypoints, ego_state, closed_loop_speed, traffic_lights):
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

    def get_state(self):
        return self._state

    def execute(self, waypoints, ego_state, closed_loop_speed, traffic_lights):
        self._state.handle(waypoints, ego_state, closed_loop_speed, traffic_lights)

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
