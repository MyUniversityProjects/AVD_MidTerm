import math

import numpy as np

import constants


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_vector(v):
    if v[0] == 0.0:
        return math.pi / 2.0
    return math.atan(v[1] / v[0])


def optimized_dist(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def get_orientation_offset(yaw, distance):
    orientation = [math.cos(yaw), math.sin(yaw)]
    return [orientation[0] * distance, orientation[1] * distance]


def move_along_orientation(pos, yaw, distance):
    offset = get_orientation_offset(yaw, distance)
    return [pos[0] + offset[0], pos[1] + offset[1]]


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


def is_vehicle_in_fov(v, ego_state, fov_angle=math.pi * 3/2):
    """Check if the vehicle is the lead vehicle."""

    vec_diff = v.pos[0] - ego_state[0], v.pos[1] - ego_state[1]
    angle = angle_between(np.array([vec_diff]), np.array(move_along_orientation((0, 0), ego_state[2], 1)))[0]

    return angle < fov_angle / 2


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

            if is_ahead and min(abs(ego_state[2] - yaw), abs(ego_state[2] + yaw)) < math.pi / 4:
                lead_car_pos.append(pos)
                lead_car_length.append(agent.vehicle.bounding_box.extent.x)
                lead_car_speed.append(agent.vehicle.forward_speed)
                lead_car_dist.append(curr_dist)
    return list(zip(lead_car_pos, lead_car_length, lead_car_speed, lead_car_dist))


def filter_neighbor_vehicles(measurement_data, ego_state):
    """Obtain Neighbor Vehicle information."""
    neighbor_car_pos = []
    neighbor_car_length = []
    neighbor_car_speed = []
    neighbor_car_dist = []
    for agent in measurement_data.non_player_agents:
        if agent.HasField('vehicle'):
            transform = agent.vehicle.transform
            pos = [transform.location.x, transform.location.y]
            yaw = math.radians(transform.rotation.yaw)

            future_pos = move_along_orientation(pos, yaw, distance=2.6)

            curr_dist = optimized_dist(pos, ego_state)
            future_dist = optimized_dist(future_pos, ego_state)
            is_ahead = curr_dist < future_dist

            if is_ahead and min(abs(ego_state[2] - yaw), abs(ego_state[2] + yaw)) < math.pi / 4:
                neighbor_car_pos.append(pos)
                neighbor_car_length.append(agent.vehicle.bounding_box.extent.x)
                neighbor_car_speed.append(agent.vehicle.forward_speed)
                neighbor_car_dist.append(curr_dist)
    return list(zip(neighbor_car_pos, neighbor_car_length, neighbor_car_speed, neighbor_car_dist))


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


def _get_ego_trajectory_points(ego_state, waypoints, closest_index):
    first_waypoint = waypoints[closest_index]
    p1, p2 = np.array(ego_state[:2]), np.array(first_waypoint[:2])
    ori = np.array([math.cos(ego_state[2]), math.sin(ego_state[2])])
    direction_vec = p2 - p1
    angle = angle_between(ori, direction_vec)
    is_behind = angle > math.pi / 2.0  # if first waypoint is behind the vehicle
    first_index = closest_index if not is_behind else closest_index + 1
    return [ego_state[:2], *waypoints[first_index:]]


def ego_pedestrian_intersect(ego_state, ped, waypoints, closest_index, ego_lookahead=10):
    """
    Checks if a pedestrian intersects with the ego vehicle trajectory.
    It creates a segment centered on the pedestrian and checks if this
    segment intersects the waypoints segments until a max distance
    (ego_lookahead).
    """
    ped_t = ped.transform
    ped_pos = np.array([ped_t.location.x, ped_t.location.y])

    points = _get_ego_trajectory_points(ego_state, waypoints, closest_index)
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

        ortho_angle = angle_vector(wp_1 - wp_2) + math.pi / 2.0
        offset_vec = np.array([math.cos(ortho_angle), math.sin(ortho_angle)]) * constants.PED_SEGMENT / 2

        if segment_intersect(wp_1, wp_2, ped_pos - offset_vec, ped_pos + offset_vec):
            return True

        len_counter += wp_len
    return False


def ego_trajectory_intersect(ego_state, ped, waypoints, closest_index, ego_lookahead=10, ped_lookahead=10, ped_speed_filter=0.02):
    """
    Checks if a pedestrian trajectory intersects with the ego vehicle trajectory.
    It creates a segment from the pedestrian to his future position and checks if this
    segment intersects the waypoints segments until a max distance (ego_lookahead).
    """
    if ped.forward_speed <= ped_speed_filter:
        return False
    ped_t = ped.transform
    ped_pos = np.array([ped_t.location.x, ped_t.location.y])
    ped_yaw = math.radians(ped_t.rotation.yaw)
    future_ped = np.array(move_along_orientation(ped_pos, ped_yaw, distance=ped_lookahead))

    points = _get_ego_trajectory_points(ego_state, waypoints, closest_index)
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

        ortho_angle = angle_vector(wp_1 - wp_2) + math.pi / 2.0
        offset_vec = np.array([math.cos(ortho_angle), math.sin(ortho_angle)]) * constants.CAR_WIDTH / 2

        for k in [-1, 1]:
            wpk_1 = wp_1 + k * offset_vec
            wpk_2 = wp_2 + k * offset_vec

            if segment_intersect(wpk_1, wpk_2, ped_pos, future_ped):
                return True

        len_counter += wp_len
    return False


def ego_vehicle_trajectory_intersect(ego_state, vehicle, closed_loop_speed, orientation_memory, waypoints, ego_lookahead=10, vehicle_lookahead=10):
    vehicle_pos = vehicle.pos[:2]
    yaw_difference = orientation_memory.get_yaw_difference(vehicle.id)
    max_iter = 1 if yaw_difference < 0.05 else 5
    cur_yaw = vehicle.yaw
    cur_pos = np.array(ego_state[:2])

    for _ in range(max_iter):
        vehicle_future_pos = move_along_orientation(vehicle_pos, cur_yaw, vehicle_lookahead)
        cur_pos = move_along_orientation(cur_pos, ego_state[2], closed_loop_speed/constants.FPS)
        cur_yaw += yaw_difference

        _, closest_index = get_closest_index(waypoints, cur_pos)
        points = _get_ego_trajectory_points([*cur_pos, cur_yaw], waypoints, closest_index)
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

            ortho_angle = angle_vector(wp_1 - wp_2) + math.pi / 2.0
            offset_vec = np.array([math.cos(ortho_angle), math.sin(ortho_angle)]) * constants.CAR_WIDTH / 2

            for k in [-1, 1]:
                wpk_1 = wp_1 + k * offset_vec
                wpk_2 = wp_2 + k * offset_vec

                if segment_intersect(wpk_1, wpk_2, vehicle_pos, vehicle_future_pos):
                    return True

            len_counter += wp_len
    return False


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


def ego_lead_intersect(ego_state, vehicles, waypoints, closed_loop_speed, ego_lookahead=0.5):
    # vehicles must be sorted by distances to ego
    ego_lookahead = ego_lookahead * closed_loop_speed if closed_loop_speed > 1 else ego_lookahead
    ego_lookahead += 15
    _, closest_index = get_closest_index(waypoints, ego_state)
    points = _get_ego_trajectory_points(ego_state, waypoints, closest_index)
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
        ortho_angle = angle_vector(wp_1 - wp_2) + math.pi / 2.0
        offset_vec = np.array([math.cos(ortho_angle), math.sin(ortho_angle)]) * constants.CAR_WIDTH
        for vehicle in vehicles:
            if segment_intersect(wp_1, wp_2, vehicle.pos - offset_vec, vehicle.pos + offset_vec):

                yaw_difference = angle_between(wp_2-wp_1, np.array(move_along_orientation((0, 0), vehicle.yaw, 1)))
                if yaw_difference < math.pi / 4.0:
                    return vehicle
        len_counter += wp_len
    return None
