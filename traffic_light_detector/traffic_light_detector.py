from enum import Enum
import itertools
import math
from pathlib import Path
from typing import *

import numpy as np

from helpers import optimized_dist
from .traffic_light_yolo_model.yolo import YOLO
from .traffic_light_ssd_model import TrafficLightModel, ModelName as SSDModelName
from .boundbox import BoundBox
import json


class TrafficLightDetector:
    # Configuration
    CONFIG_PATH = Path(__file__).parent.joinpath('traffic_light_yolo_model/config.json')
    SAVED_MODEL_PATH = Path(__file__).parent.joinpath('traffic_light_yolo_model/checkpoints/traffic-light-detection.h5')
    # Params
    TRAFFIC_SIGN_VALUE = 12
    THRESHOLD = 0.26
    SMALL_WINDOW_SIZE_THRESHOLD = 100
    ENLARGE_TARGET_AREA_RATIO = 0.0005
    AREA_RATIO_THRESHOLD = 0.20

    class ModelName(Enum):
        YOLO = True
        SSD = False

    def __init__(self, model_name: ModelName):
        self.model_name = model_name
        self.model = self.load_model(model_name)

        self._offset_w1 = self.model.MODEL_INPUT_SHAPE[0] // 2
        self._offset_h1 = self.model.MODEL_INPUT_SHAPE[1] // 2
        self._th = min(self.model.MODEL_INPUT_SHAPE) * 0.5
        self._opt_th = self._th ** 2
        self._win_size = self.model.MODEL_INPUT_SHAPE[::-1]

    def load_model(self, model_name: ModelName):
        if model_name == self.ModelName.YOLO:
            return self.load_yolo_model()
        elif model_name == self.ModelName.SSD:
            return self.load_ssd_model()

    def load_yolo_model(self):
        with open(self.CONFIG_PATH, 'r') as f:
            config = json.load(f)
        config['model']['saved_model_name'] = str(self.SAVED_MODEL_PATH)
        return YOLO(config)

    def load_ssd_model(self):
        return TrafficLightModel(SSDModelName.SSD_MOBILENET_V1)

    def detect(self, image, seg_image):
        win_images = list(self.find_semaphores(image, seg_image))
        win_images = self.merge_windows(win_images)
        win_images = self.enlarge_small_windows(image, win_images)
        images = self.extract_windows(image, win_images)
        boxes = list(map(self.predict_on_image_window, images))

        data = [(image, box) for image, box, win_image in zip(images, boxes, win_images) if box is not None and self.check_area_ratio(box, win_image, seg_image)]
        return data
        # images = [self.draw_box(*args) for args in data]
        # return list(images[i] if i < len(images) else None for i in range(3))

    def check_area_ratio(self, box, win_image, seg_image):
        start_i, end_i, start_j, end_j, _ = win_image
        h, w = end_i - start_i, end_j - start_j

        box.xmin = int((box.xmin * w) + start_j)
        box.xmax = int((box.xmax * w) + start_j)
        box.ymin = int((box.ymin * h) + start_i)
        box.ymax = int((box.ymax * h) + start_i)

        box_area = (box.xmax - box.xmin) * (box.ymax - box.ymin)
        traffic_sign_area = np.sum(seg_image[box.ymin:box.ymax, box.xmin:box.xmax] == self.TRAFFIC_SIGN_VALUE)
        ratio = traffic_sign_area / box_area
        return ratio > self.AREA_RATIO_THRESHOLD

    def enlarge_small_windows(self, image, win_images):
        h, w, *_ = image.shape
        img_size = (h, w)
        update_map = dict()

        for i, (start_i, end_i, start_j, end_j, size) in enumerate(win_images):
            if size > self.SMALL_WINDOW_SIZE_THRESHOLD:
                c = ((start_i + end_i) // 2, (start_j + end_j) // 2)
                curr_area = (end_i - start_i) * (end_j - start_j)
                target_area = size / self.ENLARGE_TARGET_AREA_RATIO
                area_multi = target_area / curr_area
                side_multi = min(2.15, math.sqrt(area_multi))

                win_size = int(self._win_size[0] * side_multi), int(self._win_size[1] * side_multi)
                curr_offset = [c[0] - start_i, c[1] - start_j]
                curr_offset[0] = int(curr_offset[0] * side_multi)
                curr_offset[1] = int(curr_offset[1] * side_multi)

                new_coords = self.get_window_coords(c, curr_offset, img_size, win_size)
                update_map[i] = (*new_coords, size)

        for i, val in update_map.items():
            win_images[i] = val

        return win_images

    def predict_on_image_window(self, image):
        boxes = self.model.predict(image)
        if self.model_name == self.ModelName.YOLO:
            boxes = [BoundBox.from_yolo_boundbox(box) for box in boxes]
        boxes = [box for box in boxes if box.score > self.THRESHOLD]
        return max(boxes, key=lambda b: b.score, default=None)

    def find_semaphores(self, image, seg_image):
        check_matrix = np.zeros(seg_image.shape, dtype=np.uint8)

        h, w, *_ = image.shape
        i_step, j_step = max(h // 100, 1), max(w // 400, 1)

        tl_value = self.TRAFFIC_SIGN_VALUE

        offset = (self._offset_h1, self._offset_w1)
        img_size = (h, w)

        min_j = w // 2       # Analyze only right elements
        max_j = w - w // 10  # Cut elements on the edge of image

        for i in range(0, h, i_step):
            for j in range(min_j, max_j, j_step):
                coord = (i, j)
                if seg_image[coord] == tl_value and check_matrix[coord] == 0:
                    cluster, center, is_rect = self.build_cluster(seg_image, coord, tl_value, check_matrix, i_step, j_step)
                    if is_rect and len(cluster) > 0:
                        start_i, end_i, start_j, end_j = self.get_window_coords(center, offset, img_size, self._win_size)
                        yield start_i, end_i, start_j, end_j, len(cluster)

    @staticmethod
    def build_cluster(seg_image, coord: Tuple[int, int], target_val, check_matrix, i_step, j_step):
        h, w, *_ = seg_image.shape
        translations = tuple((i, j) for i in [-i_step, 0, i_step] for j in [-j_step, 0, j_step] if i != 0 or j != 0)

        queue = [coord]
        cluster = list()
        while len(queue) > 0:
            point = queue.pop(0)
            if seg_image[point] == target_val:
                cluster.append(point)
                new_points = [(point[0] + t[0], point[1] + t[1]) for t in translations]
                new_points = [p for p in new_points if 0 <= p[0] < h and 0 <= p[1] < w and check_matrix[p] == 0 and seg_image[p] == target_val]
                for p in new_points:
                    check_matrix[p] = 1
                queue.extend(new_points)

        v = np.array(cluster)
        c = (np.sum(v[:, 0]) // len(v), np.sum(v[:, 1]) // len(v))

        cluster = set(cluster)

        # bot_left = (np.max(v[:, 0]), np.min(v[:, 1]))
        # inner_bot_left = (bot_left[0] - i_step, bot_left[1] + j_step)
        # is_rectangular = inner_bot_left in cluster

        # bot_point = max(v, key=lambda i: i[0])
        # bot_points = [p for p in v if abs(p[0] - bot_point[0]) <= i_step]
        # bot_left_point = min(bot_points, key=lambda i: i[1])
        # d = abs(bot_left_point[1] - bot_point[1])
        # is_rectangular = d > 40

        mini = np.min(v[:, 0])
        minj = np.min(v[:, 1])
        i_dist = abs(c[0] - mini)
        j_dist = abs(c[1] - minj) + 0.001
        shape_ratio = i_dist / j_dist
        is_rectangular = shape_ratio > 2.0

        return cluster, c, is_rectangular

    def merge_windows(self, win_images):
        win_images = np.array(win_images)
        comb_win_images = []
        indicies = set(range(len(win_images)))

        for index1, index2 in itertools.combinations(range(len(win_images)), 2):
            win1, win2 = win_images[index1], win_images[index2]
            start_i1, _, start_j1, _, size1 = win1
            p1 = (start_i1, start_j1)
            start_i2, _, start_j2, _, size2 = win2
            p2 = (start_i2, start_j2)

            d = optimized_dist(p1, p2)
            if d < self._opt_th:
                # print(f'Found One --> {d:.2f} {self._opt_th:.2f}')
                if index1 in indicies:
                    indicies.remove(index1)
                if index2 in indicies:
                    indicies.remove(index2)

                tot_size = size1 + size2
                ratio1 = size1 / tot_size
                ratio2 = size2 / tot_size

                start_i = int(p1[0] * ratio1 + p2[0] * ratio2)
                start_j = int(p1[1] * ratio1 + p2[1] * ratio2)

                end_i = start_i + self._win_size[0]
                end_j = start_j + self._win_size[1]
                comb_win_images.append((start_i, end_i, start_j, end_j, tot_size))

        return comb_win_images + list(win_images[list(indicies)])

    @staticmethod
    def extract_windows(src_image, win_images):
        return [src_image[start_i:end_i, start_j:end_j, ...] for start_i, end_i, start_j, end_j, _ in win_images]

    def draw_box(self, image, box, border=2):
        image_h, image_w, _ = image.shape
        if box.xmin > image_w or box.ymin > image_h:
            return
        res = image

        # xmin = int(box.xmin * image_w)
        # ymin = int(box.ymin * image_h)
        # xmax = int(box.xmax * image_w)
        # ymax = int(box.ymax * image_h)

        xmin = box.xmin
        ymin = box.ymin
        xmax = box.xmax
        ymax = box.ymax

        xmin = max(xmin, 0)
        xmax = self.clamp(xmax, 0, image_w - 1)
        ymin = max(ymin, 0)
        ymax = self.clamp(ymax, 0, image_h - 1)

        color = box.color

        res[ymin:ymax, xmin:xmin+border] = np.array(color)
        res[ymin:ymax, xmax:xmax+border] = np.array(color)
        res[ymin:ymin+border, xmin:xmax] = np.array(color)
        res[ymax:ymax+border, xmin:xmax] = np.array(color)

        return res

    @staticmethod
    def get_window_coords(center, offset, img_size, win_size):
        c_i, c_j = center
        offset_h, offset_w = offset
        win_h, win_w = win_size
        h, w = img_size

        start_i = max(c_i - offset_h, 0)
        overflow = max(start_i + win_h - h, 0)
        start_i -= overflow
        end_i = start_i + win_h

        start_j = max(c_j - offset_w, 0)
        overflow = max(start_j + win_w - w, 0)
        start_j -= overflow
        end_j = start_j + win_w

        return start_i, end_i, start_j, end_j

    @staticmethod
    def clamp(val, min_val, max_val):
        return max(min(val, max_val), min_val)
