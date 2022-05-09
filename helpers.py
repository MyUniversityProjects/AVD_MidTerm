import math

import numpy as np


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def optimized_dist(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def get_orientation_offset(yaw, distance):
    orientation = [math.cos(yaw), math.sin(yaw)]
    return [orientation[0] * distance, orientation[1] * distance]


def move_along_orientation(pos, yaw, distance):
    offset = get_orientation_offset(yaw, distance)
    return [pos[0] + offset[0], pos[1] + offset[1]]


def triangle_area(a, b, c):
    return abs((b[0] * a[1] - a[0] * b[1]) + (c[0] * b[1] - b[0] * c[1]) + (a[0] * c[1] - c[0] * a[1])) / 2


def get_test_rect_area(rect, point):
    a, b, c, d = rect
    return triangle_area(a, b, point) + triangle_area(a, d, point) + triangle_area(b, c, point) + triangle_area(d, c, point)


def point_on_segment(p1, p2, p3):
    return p2[0] <= max(p1[0], p3[0]) and (p2[0] >= min(p1[0], p3[0])) and (p2[1] <= max(p1[1], p3[1])) and (p2[1] >= min(p1[1], p3[1]))


def segment_intersect(p11, p12, p21, p22):
    v1 = np.subtract(p12, p11)
    v2 = np.subtract(p21, p12)
    sign_1 = np.sign(np.cross(v1, v2))
    v2 = np.subtract(p22, p12)
    sign_2 = np.sign(np.cross(v1, v2))

    v1 = np.subtract(p22, p21)
    v2 = np.subtract(p11, p22)
    sign_3 = np.sign(np.cross(v1, v2))
    v2 = np.subtract(p12, p22)
    sign_4 = np.sign(np.cross(v1, v2))

    # Check if the line segments intersect.
    if (sign_1 != sign_2) and (sign_3 != sign_4):
        return True

    # Check if the collinearity cases hold.
    if sign_1 == 0 and point_on_segment(p11, p21, p12):
        return True
    if sign_2 == 0 and point_on_segment(p11, p22, p12):
        return True
    if sign_3 == 0 and point_on_segment(p21, p11, p22):
        return True
    if sign_3 == 0 and point_on_segment(p21, p12, p22):
        return True

    return False


def filter_lead_vehicle_orientation(measurement_data, ego_state):
    """Obtain Lead Vehicle information."""

    lead_car_pos = []
    lead_car_length = []
    lead_car_speed = []
    lead_car_dist = []
    for agent in measurement_data.non_player_agents:
        if agent.HasField('vehicle'):
            transform = agent.vehicle.transform
            pos = [transform.location.x, transform.location.y]
            yaw = math.radians(transform.rotation.yaw)

            future_pos = move_along_orientation(pos, yaw, distance=2.6)

            curr_dist = optimized_dist(pos, ego_state)
            future_dist = optimized_dist(future_pos, ego_state)
            is_ahead = curr_dist < future_dist

            if is_ahead and abs(ego_state[2] - yaw) < math.pi / 4:  # TODO: considerare
                lead_car_pos.append(pos)
                lead_car_length.append(agent.vehicle.bounding_box.extent.x)
                lead_car_speed.append(agent.vehicle.forward_speed)
                lead_car_dist.append(curr_dist)
    return list(zip(lead_car_pos, lead_car_length, lead_car_speed, lead_car_dist))


def check_for_path_intersection(waypoints, closest_index, goal_index, agents):
    """
    Checks for an agent that is intervening the goal path.

    Checks for an agent that is intervening the goal path. Returns a new
    goal index and the index of the agent obstruction found.

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
            agent_index: index of the agent found; -1 otherwise
    """
    for i in range(closest_index, goal_index):
        # Check to see if path segment crosses any of the stop lines.
        for index, agent in enumerate(agents):
            wp_1 = np.array(waypoints[i][0:2])
            wp_2 = np.array(waypoints[i + 1][0:2])
            sl_1 = np.array(agent[0:2])
            sl_2 = np.array(agent[2:4])

            # If there is an intersection, update
            # the goal state to stop before the goal line.
            if segment_intersect(wp_1, wp_2, sl_1, sl_2):
                goal_index = i
                return goal_index, index
    return goal_index, -1


def pedestrian_is_ahead(ego_state, ped, lookahead):
    """
    Checks if a pedestrian is ahead of the vehicle.
    It creates a rectangle as bounding box that has:
        - Long axis with the same orientation of the vehicle
        - Center equal to the pedestrian position
    """
    lookahead /= 2
    rect_width = 1.5
    rect_area = lookahead * 2 * rect_width * 2

    loc = ped.transform.location
    ped_pos = np.array([loc.x, loc.y])

    yaw = ego_state[2]
    future_ego_state = move_along_orientation(ego_state, yaw, distance=lookahead)

    v1 = np.array(get_orientation_offset(yaw, distance=lookahead))
    v2 = np.array(get_orientation_offset(yaw - math.pi / 2, distance=rect_width))

    a = ped_pos + v1 + v2
    b = ped_pos + v1 - v2
    c = ped_pos - v1 - v2
    d = ped_pos - v1 + v2
    rect = [a, b, c, d]

    return get_test_rect_area(rect, future_ego_state) <= rect_area


def ego_trajectory_intersect(ego_state, ped, waypoints, closest_index, ego_lookahead=10, ped_lookahead=10, ped_speed_filter=0.02):
    """ TODO: change
    Checks if a pedestrian trajectory intersects with the ego vehicle trajectory.
    It creates segment trajectory using objects position and translating it
    along the object orientation by an amount specified by the lookahead parameter
    """
    if ped.forward_speed <= ped_speed_filter:
        return False
    ped_t = ped.transform
    ped_pos = np.array([ped_t.location.x, ped_t.location.y])
    ped_yaw = ped_t.rotation.yaw
    future_ped = np.array(move_along_orientation(ped_pos, ped_yaw, distance=ped_lookahead))

    first_waypoint = waypoints[closest_index]
    p1, p2 = np.array(ego_state[:2]), np.array(first_waypoint[:2])
    ori = np.array([math.cos(ego_state[2]), math.sin(ego_state[2])])
    direction_vec = p2 - p1
    angle = angle_between(ori, direction_vec)
    is_behind = angle > math.pi / 2.0  # if first waypoint is behind the vehicle
    first_index = closest_index if not is_behind else closest_index + 1

    points = [ego_state[:2], *waypoints[first_index:]]
    len_counter = 0

    for i in range(len(points) - 1):
        if len_counter >= ego_lookahead:
            break
        wp_1 = np.array(points[i][0:2])
        wp_2 = np.array(points[i + 1][0:2])

        wp_len = math.sqrt(optimized_dist(wp_1, wp_2))
        if wp_len + len_counter >= ego_lookahead:
            len_remaining = ego_lookahead - len_counter
            centered_wp_2 = wp_2 - wp_1
            wp_2 = (centered_wp_2 / np.linalg.norm(centered_wp_2) * len_remaining) + wp_1

        if segment_intersect(wp_1, wp_2, ped_pos, future_ped):
            return True

        len_counter += wp_len
    return False


def is_ahead(p1, p2, yaw):
    p1, p2 = np.array(p1), np.array(p2)
    ori = np.array([math.cos(yaw), math.sin(yaw)])
    direction_vec = p2 - p1
    return angle_between(ori, direction_vec) < math.pi / 2.0
