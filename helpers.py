import math

import numpy as np


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

            if is_ahead and abs(ego_state[2] - yaw) < math.pi / 4:
                lead_car_pos.append(pos)
                lead_car_length.append(agent.vehicle.bounding_box.extent.x)
                lead_car_speed.append(agent.vehicle.forward_speed)
                lead_car_dist.append(curr_dist)
    return list(zip(lead_car_pos, lead_car_length, lead_car_speed, lead_car_dist))


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
