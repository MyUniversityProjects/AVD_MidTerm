import numpy as np
import pygame
import time
from carla import image_converter


class Interface:
    WINDOW_WIDTH = 832
    WINDOW_HEIGHT = 832

    def __init__(self, render_interval: float = 1):
        self.render_interval = render_interval
        self.last_render = time.time()
        pygame.init()
        self._display = pygame.display.set_mode(
            (self.WINDOW_WIDTH, self.WINDOW_HEIGHT),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.base_image = np.zeros((self.WINDOW_HEIGHT, self.WINDOW_WIDTH, 3))

    def show_images(self, sensor_data):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        curr_time = time.time()
        if curr_time - self.last_render < self.render_interval:
            return
        self.last_render = curr_time

        image = self.base_image

        rgb_image = sensor_data.get('CameraRGB', None)
        if rgb_image is not None:
            image[:416, :416, :] = image_converter.to_rgb_array(rgb_image)

        seg_image = sensor_data.get('CameraSemanticSegmentation', None)
        if seg_image is not None:
            image[:416, 416:, :] = image_converter.labels_to_cityscapes_palette(seg_image)

        depth_image = sensor_data.get('CameraDepth', None)
        if depth_image is not None:
            image[416:, :416, :] = image_converter.depth_to_logarithmic_grayscale(depth_image)

        surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        self._display.blit(surface, (0, 0))

        pygame.display.flip()
