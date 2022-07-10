from typing import List

import dataclasses
import numpy as np

from helpers import optimized_dist
from .traffic_light_detector import TrafficLightDetector
from .boundbox import BoundBox


@dataclasses.dataclass
class TrackerObject:
    box: BoundBox
    ttl: int
    tte: int


class TrafficLightTracker(TrafficLightDetector):
    TTL = 5
    TTE = 3
    MAX_DIST = 160 ** 2
    THRESHOLD = 0.1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._objects: List[TrackerObject] = []

    def detect(self, image, seg_image):
        print(self._objects)
        # 1 - Retrieve new blobs
        blobs = super().detect(image, seg_image)
        if len(blobs) == 0 and len(self._objects) == 0:
            return image
        # 2 - Compute similarity matrix
        similarity_mat = self._build_similarity_matrix(blobs)
        # 3 - Unmark blobs and objects
        unassociated_blobs = set(range(len(blobs)))
        unassociated_objects = set(range(len(self._objects)))
        # 4 - Associate blobs with objects
        if len(blobs) > 0 and len(self._objects) > 0:
            while np.max(similarity_mat) > self.THRESHOLD:
                i, j = np.unravel_index(np.argmax(similarity_mat), shape=similarity_mat.shape)
                blob, obj = blobs[i], self._objects[j]
                self._update_object(blob, obj)
                similarity_mat[i, :] = 0
                similarity_mat[:, j] = 0
                unassociated_blobs.remove(i)
                unassociated_objects.remove(j)
        # 5 - Update exiting objects
        for j in unassociated_objects:
            obj = self._objects[j]
            obj.ttl -= 1
        self._objects = [o for o in self._objects if o.ttl > 0]
        # 6 - Create new objects
        for i in unassociated_blobs:
            blob = blobs[i]
            obj = TrackerObject(box=blob[1], ttl=self.TTL, tte=1)
            self._objects.append(obj)
        # Draw boxes
        actual_objects = [o for o in self._objects if o.tte >= self.TTE]
        if len(actual_objects) > 0:
            image = image.copy()
        for o in actual_objects:
            image = super().draw_box(image, o.box)
        return image

    def _update_object(self, blob, obj):
        obj.ttl = self.TTL  # Reset object TTL
        obj.tte = min(obj.tte + 1, self.TTE)
        obj.box = blob[1]

    def _build_similarity_matrix(self, blobs):
        similarity_mat = np.zeros((len(blobs), len(self._objects)))
        for i, blob in enumerate(blobs):
            for j, obj in enumerate(self._objects):
                similarity_mat[i, j] = self._get_similarity(blob, obj)
        return similarity_mat

    @classmethod
    def _get_similarity(cls, blob, obj: TrackerObject):
        _, blob_box = blob
        blob_c = blob_box.center
        obj_c = obj.box.center
        dist = optimized_dist(blob_c, obj_c)

        return max(1 - dist / cls.MAX_DIST, 0)
