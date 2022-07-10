from pathlib import Path

import h5py
import numpy as np


class Recorder:
    MAX_FRAMES = 120
    SAVE_PATH = Path(__file__).parent.joinpath('frames.h5')

    def __init__(self):
        self._bucket = ([], [])
        self._bucket_i = 0
        self._ds_i = 1

    def save_frame(self, rgb_image, seg_image):
        self._bucket[0].append(rgb_image)
        self._bucket[1].append(seg_image)
        self._bucket_i = (self._bucket_i + 1) % self.MAX_FRAMES
        if self._bucket_i == 0:
            print(f'Saving {len(self._bucket[0])} frames bucket')
            with h5py.File(self.SAVE_PATH, 'a') as f:
                f.create_dataset(f'rgb{self._ds_i}', data=np.array(self._bucket[0]))
                f.create_dataset(f'seg{self._ds_i}', data=np.array(self._bucket[1]))
                self._bucket = ([], [])
                self._ds_i += 1

    def iter_frames(self, frames_path=None, jump=0):
        if frames_path is None:
            frames_path = self.SAVE_PATH
        ds_i = 1
        with h5py.File(frames_path, 'r') as f:
            while jump > 0:
                rgb_key, seg_key = f'rgb{ds_i}', f'seg{ds_i}'
                if rgb_key not in f.keys() or seg_key not in f.keys():
                    break
                jump -= f[rgb_key].shape[0]
                ds_i += 1

            while True:
                rgb_key, seg_key = f'rgb{ds_i}', f'seg{ds_i}'
                if rgb_key not in f.keys() or seg_key not in f.keys():
                    break
                for rgb_img, seg_img in zip(f[rgb_key], f[seg_key]):
                    yield rgb_img, seg_img
                ds_i += 1
