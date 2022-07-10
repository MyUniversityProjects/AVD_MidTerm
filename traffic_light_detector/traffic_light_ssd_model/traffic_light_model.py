from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
import os
import tensorflow as tf
from ..boundbox import BoundBox

tf.compat.v1.disable_eager_execution()
# from PIL import Image, ImageDraw as D


class ModelName(Enum):
    SSD_MOBILENET_V1 = 'ssd_mobilenet_v1'
    SSD_MOBILENET_V2 = 'ssd_mobilenet_v2'
    SSDLITE_MOBILENET_V2 = 'ssdlite_mobilenet_v2'
    SSD_INCEPTION_V2 = 'ssd_inception_v2'


class TrafficLightModel:
    def __init__(self, model_name: ModelName):
        self._model_name = model_name
        self._model_path = TrafficLightModel.build_model_path(model_name)
        self._graph = TrafficLightModel.load_model(self._model_path)

    def _run_inference(self, sess, ops, image_tensor, image) -> List[BoundBox]:
        # output_dict = {}

        # time_s = time.time()
        num_detections, boxes, scores, classes = sess.run(ops, feed_dict={image_tensor: image})
        # time_t = time.time() - time_s

        # output_dict['num_detections'] = int(num_detections[0])
        # output_dict['detection_classes'] = classes[0].astype(np.uint8)
        # output_dict['detection_boxes'] = boxes[0]
        # output_dict['detection_scores'] = scores[0]
        # output_dict['detection_time'] = time_t

        detection_classes = classes[0].astype(np.uint8)
        detection_boxes = [(box[1], box[0], box[3], box[2]) for box in boxes[0]]
        detection_scores = scores[0]

        return [BoundBox(*args) for args in zip(detection_boxes, detection_classes, detection_scores)]

    def predict(self, image: np.ndarray):
        with self._graph.as_default():
            image_tensor = self._graph.get_tensor_by_name('image_tensor:0')
            boxes_tensor = self._graph.get_tensor_by_name('detection_boxes:0')
            scores_tensor = self._graph.get_tensor_by_name('detection_scores:0')
            classes_tensor = self._graph.get_tensor_by_name('detection_classes:0')
            detections_tensor = self._graph.get_tensor_by_name('num_detections:0')

            ops = [detections_tensor, boxes_tensor, scores_tensor, classes_tensor]

            with tf.compat.v1.Session() as sess:
                image_np_expanded = np.expand_dims(image, axis=0)
                # Actual detection.
                return self._run_inference(sess, ops, image_tensor, image_np_expanded)

    @staticmethod
    def load_model(model_path: str) -> tf.Graph:
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    @staticmethod
    def build_model_path(model_name):
        return str(Path(__file__).parent.joinpath(f'models/{model_name.value}/frozen_inference_graph.pb'))
        return os.path.join('models', model_name.value, "frozen_inference_graph.pb")



