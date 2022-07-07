from pathlib import Path
from .traffic_light_model.yolo import YOLO
import json


class TrafficLightDetector:
    CONFIG_PATH = Path(__file__).parent.joinpath('traffic_light_model/config.json')
    SAVED_MODEL_PATH = Path(__file__).parent.joinpath('traffic_light_model/checkpoints/traffic-light-detection.h5')

    def __init__(self):
        with open(self.CONFIG_PATH, 'r') as f:
            config = json.load(f)
        config['model']['saved_model_name'] = str(self.SAVED_MODEL_PATH)
        self.model = YOLO(config)

    def detect(self, image):
        boxes = self.model.predict(image)
        return 'HeHe'
