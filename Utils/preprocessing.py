import numpy as np
import cv2
from skimage import exposure
from fastai.vision import Image, get_transforms, image2np


class Preprocessing:

    def __init__(self, enhance_contrast=True, median_blur=True, blur_kernel=5):

        self.enhance_contrast = enhance_contrast
        self.median_blur = median_blur
        self.blur_kernel = blur_kernel

    def __call__(self, inp):

        if self.enhance_contrast:
            inp = cv2.equalizeHist(inp)
            # inp = exposure.equalize_adapthist(inp)
            # inp = np.clip(inp * 255., 0, 255).astype(np.uint8)

        if self.median_blur:
            inp = cv2.medianBlur(inp, 5 if self.blur_kernel is None else self.blur_kernel)

        return inp
