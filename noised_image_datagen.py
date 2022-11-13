import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from img_utils import ImgUtils
from noise_scheduler import NoiseScheduler


class NoisedImageDatagen(tf.keras.utils.Sequence):
    def __init__(self, img_fps, img_size=(64,64), n_timesteps=50, noise_start=0.0001, noise_end=0.06):
        imgs = [plt.imread(fn) for fn in img_fps]
        imgs = [ImgUtils.resize_img(img, img_size) for img in imgs]
        imgs = [ImgUtils.int_to_float_img(img) for img in imgs]
        self.imgs = [ImgUtils.scale_img(img) for img in imgs]
        self.n_timesteps = n_timesteps
        self.ns = NoiseScheduler(n_timesteps, noise_start, noise_end)

    def __len__(self):  # batches per epoch
        return len(self.imgs)

    def __getitem__(self, index):  # fetch one batch of data
        batch_X = np.array([self.ns.forward(self.imgs[index], step)
                           for step in range(self.n_timesteps)])
        batch_y = np.array([self.imgs[index] - batch_X[nstep]
                           for nstep in range(self.n_timesteps)])
        return batch_X, batch_y
