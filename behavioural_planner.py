#!/usr/bin/env python3
import logging

import numpy as np
import math

from behavioural_states import StateManager, DecelerateToPointState, EmergencyState, StopState
from custom_agents import OrientationMemory

# Stop speed threshold
STOP_THRESHOLD = 0.02


class BehaviouralPlanner:
    def __init__(self, lookahead, lead_vehicle_lookahead):
        self._lookahead                     = lookahead
        self._follow_lead_vehicle_lookahead = lead_vehicle_lookahead
        self._orientation_memory            = OrientationMemory()
        self._state_manager                 = StateManager(self)
        self._follow_lead_vehicle           = False
        self._obstacle_on_lane              = False
        self._goal_state                    = [0.0, 0.0, 0.0]
        self._goal_index                    = 0
        self._stop_count                    = 0
        self._lookahead_collision_index     = 0
        self._traffic_light_id              = None
        self._visited_agents                = set()  # for eventual stop sign
        self._emergency_brake_value         = 0.0

    def get_lookahead(self):
        return self._lookahead

    def set_lookahead(self, lookahead):
        self._lookahead = lookahead

    def get_orientation_memory(self):
        return self._orientation_memory

    def get_traffic_light_id(self):
        return self._traffic_light_id

    def set_traffic_light_id(self, tl_id):
        self._traffic_light_id = tl_id

    def get_visited_agents(self):
        return self._visited_agents

    def get_goal_state(self):
        return self._goal_state

    def set_goal_state(self, goal_state):
        self._goal_state = goal_state

    def get_goal_index(self):
        return self._goal_index

    def set_goal_index(self, goal_index):
        self._goal_index = goal_index

    def set_following_lead_vehicle(self, val):
        if val != self._follow_lead_vehicle:
            if val == True:
                logging.info('Following lead vehicle')
            else:
                logging.info('Stop following lead vehicle')
            self._follow_lead_vehicle = val

    def get_emergency_brake_value(self):
        return self._emergency_brake_value

    def set_emergency_brake_value(self, brake_value):
        self._emergency_brake_value = brake_value

    def in_emergency(self):
        return self._state_manager.get_state().NAME == EmergencyState.NAME

    def in_stop(self):
        return self._state_manager.get_state().NAME == StopState.NAME

    def is_decelerating(self):
        return self._state_manager.get_state().NAME == DecelerateToPointState.NAME

    # Handles state transitions and computes the goal state.
    def transition_state(self, waypoints, ego_state, closed_loop_speed, pedestrians, vehicles, traffic_lights):
        """Handles state transitions and computes the goal state.  
        
        args:
            waypoints: current waypoints to track (global frame). 
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
            closed_loop_speed: current (closed-loop) speed for vehicle (m/s)
        variables to set:
            self._goal_index: Goal index for the vehicle to reach
                i.e. waypoints[self._goal_index] gives the goal waypoint
            self._goal_state: Goal state for the vehicle to reach (global frame)
                format: [x_goal, y_goal, v_goal]
            self._state: The current state of the vehicle.
                available states: 
                    FOLLOW_LANE         : Follow the global waypoints (lane).
                    DECELERATE_TO_STOP  : Decelerate to stop.
                    STAY_STOPPED        : Stay stopped.
            self._stop_count: Counter used to count the number of cycles which
                the vehicle was in the STAY_STOPPED state so far.
        useful_constants:
            STOP_THRESHOLD  : Stop speed threshold (m). The vehicle should fully
                              stop when its speed falls within this threshold.
            STOP_COUNTS     : Number of cycles (simulation iterations) 
                              before moving from stop sign.
        """
        self._emergency_brake_value = 0.0
        self._state_manager.execute(waypoints, ego_state, closed_loop_speed, pedestrians, vehicles, traffic_lights)
