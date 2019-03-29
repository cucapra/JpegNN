from PIL import Image, ImageFilter
import cv2
import numpy as np

class Interpolate(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        cvimg = np.asarray(sample)
        cvimg=cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR) 
        resized = cv2.resize(cvimg, (self.output_size,self.output_size),interpolation=cv2.INTER_AREA )
        pilimg = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return  Image.fromarray(pilimg)
