import math


def optimized_dist(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


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

            orientation = [math.cos(yaw) * 2.6, math.sin(yaw) * 2.6]
            future_pos = [pos[0] + orientation[0], pos[1] + orientation[1]]

            curr_dist = optimized_dist(pos, ego_state)
            future_dist = optimized_dist(future_pos, ego_state)
            is_lead = curr_dist < future_dist

            if is_lead and abs(ego_state[2] - yaw) < math.pi / 4:
                lead_car_pos.append(pos)
                lead_car_length.append(agent.vehicle.bounding_box.extent.x)
                lead_car_speed.append(agent.vehicle.forward_speed)
                lead_car_dist.append(curr_dist)
    return list(zip(lead_car_pos, lead_car_length, lead_car_speed, lead_car_dist))
