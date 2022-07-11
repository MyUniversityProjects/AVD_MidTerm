class BoundBox:
    YOLO_CLASS_TO_COLOR = {
        0: (0, 255, 0),
        1: (255, 0, 0),
    }
    SSD_CLASS_TO_COLOR = {
        1: (0, 255, 0),
        2: (255, 255, 0),
        3: (255, 0, 0),
    }

    def __init__(self, box_coords, pred_class, score):
        self.xmin = box_coords[0]
        self.ymin = box_coords[1]
        self.xmax = box_coords[2]
        self.ymax = box_coords[3]

        self.pred_class = pred_class

        self.label = pred_class
        self.score = score
        self._center = None
        self.from_yolo = False

    @staticmethod
    def from_yolo_boundbox(box):
        box_coords = (box.xmin, box.ymin, box.xmax, box.ymax)
        box = BoundBox(box_coords, box.get_label(), box.c)
        box.from_yolo = True
        return box

    @property
    def color(self):
        if self.from_yolo:
            return self.YOLO_CLASS_TO_COLOR[self.pred_class]
        return self.SSD_CLASS_TO_COLOR[self.pred_class]

    @property
    def center(self):
        if self._center is None:
            cx = (self.xmin + self.xmax) / 2
            cy = (self.ymin + self.ymax) / 2
            self._center = cx, cy
        return self._center

    def __str__(self):
        c = self.center
        return f'({c[0]}, {c[1]}, {self.xmin}, {self.ymin}, {self.xmax}, {self.ymax})'

    def __repr__(self):
        return str(self)
