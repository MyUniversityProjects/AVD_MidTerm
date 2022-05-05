class TrafficLightAdapter:
    def __init__(self, agent):
        self.id = agent.id
        tl_t = agent.traffic_light.transform
        self.position = [tl_t.location.x, tl_t.location.y, tl_t.location.z]
        self.yaw = tl_t.rotation.yaw
    