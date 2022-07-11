from pathlib import Path

import numpy as np

from interface import Interface
from traffic_light_detector import TrafficLightDetector, TrafficLightTracker
from utils import Recorder

CLASSES = {
    0: [0, 0, 0],  # None
    1: [70, 70, 70],  # Buildings
    2: [190, 153, 153],  # Fences
    3: [72, 0, 90],  # Other
    4: [220, 20, 60],  # Pedestrians
    5: [153, 153, 153],  # Poles
    6: [157, 234, 50],  # RoadLines
    7: [128, 64, 128],  # Roads
    8: [244, 35, 232],  # Sidewalks
    9: [107, 142, 35],  # Vegetation
    10: [0, 0, 255],  # Vehicles
    11: [102, 102, 156],  # Walls
    12: [220, 220, 0]  # TrafficSigns
}

IMG_SIZE = 1000
SHOW_INTERVAL = 0.001
JUMP_FRAMES = 120 * 12
FRAMES_PATH = Path(__file__).parent.joinpath('frames.h5')


def get_rgb_seg_image(array):
    result = np.zeros((array.shape[0], array.shape[1], 3))
    for key, value in CLASSES.items():
        result[np.where(array == key)] = value
    return result


def main():
    interface = Interface(render_interval=SHOW_INTERVAL, slots=1, img_size=(IMG_SIZE, IMG_SIZE))
    tl_detector = TrafficLightTracker(TrafficLightDetector.ModelName.YOLO)
    recorder = Recorder()

    for rgb_img, seg_img in recorder.iter_frames(jump=JUMP_FRAMES):
        rgb_image = tl_detector.detect(rgb_img, seg_img)
        interface.show_images(images=[rgb_image])

    print('Done.')


if __name__ == '__main__':
    main()
