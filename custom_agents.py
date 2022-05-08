import numpy as np


class TrafficLightAdapter:
    GREEN = 0
    TRAFFIC_LIGHT_LENGTH = 5

    def __init__(self, agent):
        self.agent = agent
        self.id = agent.id
        tl_t = agent.traffic_light.transform
        self.position = [tl_t.location.x, tl_t.location.y, tl_t.location.z, tl_t.rotation.yaw]
        self.yaw = tl_t.rotation.yaw
        self.state = agent.traffic_light.state

    def get_segment(self):
        x, y, z, yaw = self.position
        abs_yaw = abs(yaw)
        sign = -1 if abs_yaw < 45 or abs_yaw > 90 + 45 else 1
        yaw = (yaw * np.pi / 180.0) + (sign * np.pi / 2.0)
        spos = np.array([
            [0, 0],
            [0, self.TRAFFIC_LIGHT_LENGTH],
        ])
        rotyaw = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)],
        ])
        spos_shift = np.array([
            [x, x],
            [y, y],
        ])
        spos = np.add(np.matmul(rotyaw, spos), spos_shift)
        return [spos[0, 0], spos[1, 0], spos[0, 1], spos[1, 1]]

    def __str__(self):
        return f'TrafficLight(id={self.id}, position={self.position}, yaw={self.yaw}, state={self.state})'

    def __repr__(self):
        return str(self)
