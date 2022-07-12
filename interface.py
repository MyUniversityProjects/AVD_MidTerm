import cv2
import numpy as np
import pygame
import time
# from carla import image_converter


class Interface:
    BOX_SIZE = 416

    def __init__(self, render_interval: float = 1, show=True, slots=1, img_size=None):
        self.show = show
        self.img_w, self.img_h = self.BOX_SIZE, self.BOX_SIZE
        if img_size is not None:
            self.img_w, self.img_h = img_size

        self.win_width = self.img_w if slots < 2 else self.img_w * 2
        self.win_height = self.img_h if slots < 3 else self.img_h * 2

        self.render_interval = render_interval
        self.last_render = time.time()
        pygame.init()
        self._display = pygame.display.set_mode((self.win_width, self.win_height), pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.base_image = np.zeros((self.win_height, self.win_width, 3))

    def show_images(self, images):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        if not self.show or len([i for i in images if i is not None]) == 0:
            return

        curr_time = time.time()
        if curr_time - self.last_render < self.render_interval:
            return
        self.last_render = curr_time

        image = self.base_image

        start_i, start_j = 0, 0
        for img in images:
            if img is not None:
                h, w, *_ = img.shape
                if h != self.img_h or w != self.img_w:
                    img = cv2.resize(img, (self.img_h, self.img_w))
                image[start_i:start_i+self.img_h, start_j:start_j+self.img_w, :] = img
            start_i, start_j = self._next_starts(start_i, start_j, self.img_h, self.img_w)

        # if self.show_depth:
        #     if depth_image is not None:
        #         image[start_i:start_i+self.BOX_SIZE, start_j:start_j+self.BOX_SIZE, :] = image_converter.depth_to_logarithmic_grayscale(depth_image)

        surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        self._display.blit(surface, (0, 0))

        pygame.display.flip()

    def _next_starts(self, i, j, height, width):
        j += width
        if j >= self.win_width:
            j = 0
            i += height
        return i, j
