import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from .boundbox import BoundBox


class BoundBoxKalmanFilter:
    def __init__(self, box: BoundBox, dt=1, std_acc=10, std_pred=40, std_meas=40):
        self._box = box
        self._dt = dt
        self._std_acc = std_acc
        self._std_pred = std_pred
        self._std_meas = std_meas

        self._f = self._create_kalman_filter()

        self._old_measure = None
        self._new_measure = None

    @property
    def old_measure(self):
        return self._old_measure

    def _create_kalman_filter(self):
        c = self._box.center

        f = KalmanFilter(dim_x=4, dim_z=2)

        f.x = np.array([c[0], c[1], 0.0, 0.0])

        f.F = np.array([[1.0, 0, self._dt, 0],
                        [0, 1.0, 0, self._dt],
                        [0, 0, 1.0, 0],
                        [0, 0, 0, 1.0]])

        f.H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])

        # f.B = np.matrix([[(self._dt**2) / 2, 0],
        #                 [0, (self._dt**2) / 2],
        #                 [self._dt, 0],
        #                 [0, self._dt]])

        f.P *= self._std_pred ** 2

        f.R *= self._std_meas ** 2

        f.Q = Q_discrete_white_noise(dim=4, dt=self._dt, var=self._std_acc ** 2)

        return f

    def predict(self, box: BoundBox = None):
        self._old_measure = self._new_measure

        z = self._new_measure  # use prediction as input when box is None
        if box is not None:
            self._box = box
            z = box.center

        self._f.predict()
        self._f.update(np.array(z).reshape((2, 1)))

        self._new_measure = self._f.measurement_of_state(self._f.x)
        return self._new_measure
