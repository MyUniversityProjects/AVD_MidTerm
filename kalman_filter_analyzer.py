import itertools
import json
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from traffic_light_detector.BoundBoxKalmanFilter import BoundBoxKalmanFilter
from traffic_light_detector.boundbox import BoundBox


MEASURES_FILE_PATH = 'utils/measures.json'


def load_data():
    with open(MEASURES_FILE_PATH, 'r') as f:
        data = json.load(f)
    return np.array(data)


def data2box(d):
    return BoundBox(d[-4:], pred_class=-1, score=-1)


def grid_search(data, dt_list, std_acc_list, std_pred_list, std_meas_list):
    boxes = [data2box(sample) for sample in data]

    mse_array = np.zeros((len(dt_list) * len(std_acc_list) * len(std_pred_list) * len(std_meas_list), 5))

    fist_box = boxes[0]
    boxes = boxes[1:]

    params = list(itertools.product(dt_list, std_acc_list, std_pred_list, std_meas_list, repeat=1))
    for i, (dt, std_acc, std_pred, std_meas) in tqdm(list(enumerate(params))):
        try:
            kf = BoundBoxKalmanFilter(fist_box, dt=dt, std_acc=std_acc, std_pred=std_pred, std_meas=std_meas)
            pred_array = np.array([kf.predict(box) for box in boxes])[:-1]

            errors_x = pred_array[:, 0] - data[2:, 0]
            mse_x = np.mean(errors_x ** 2)

            errors_y = pred_array[:, 1] - data[2:, 1]
            mse_y = np.mean(errors_y ** 2)

            err = mse_x + mse_y

            mse_array[i] = (dt, std_acc, std_pred, std_meas, err)
        except:
            max_val = np.finfo(mse_array.dtype).max
            mse_array[i] = (max_val, max_val, max_val, max_val, max_val)
            print('Jumping:', (dt, std_acc, std_pred, std_meas))

    return mse_array


array = load_data()

# dt_choices = [0.5, 1, 2, 16, 32, 64]
# choices = [0.01, 0.1, 1, 2, 5, 1280, 1280*2, 1280*4]
# res = grid_search(array, dt_choices, choices, choices, choices)
# ordered_res = sorted(res, key=lambda e: e[-1])
# print(*ordered_res[:15], sep='\n')


# plt.plot(np.arange(array.shape[0]), array[:, 0], array[:, 1])
# plt.show()
# plt.plot(np.arange(array.shape[0]), array[:, 2], array[:, 3])
# plt.show()

vals_x = np.array([((e1 - e2) ** 2) for e1, e2 in zip(array[:, 0], array[1:, 0])])
vals_y = np.array([((e1 - e2) ** 2) for e1, e2 in zip(array[:, 1], array[1:, 1])])
print(np.mean(vals_x) + np.mean(vals_y))
# plt.plot(vals)
# plt.show()


kf = BoundBoxKalmanFilter(data2box(array[0]), dt=16, std_acc=2560, std_pred=0.1, std_meas=1)
pred_array = np.array([kf.predict(data2box(p)) for p in array[1:]])

plt.plot(np.arange(pred_array.shape[0] - 1), pred_array[:-1, 1], array[2:, 1])
plt.legend(['pred', 'real'])
plt.show()
plt.plot(np.arange(pred_array.shape[0] - 1), pred_array[:-1, 0], array[2:, 0])
plt.legend(['pred', 'real'])
plt.show()

errors = pred_array[:-1, 1] - array[2:, 1]
print(np.min(errors), np.max(errors), np.mean(errors ** 2))
#
# plt.plot(errors)
# plt.show()

