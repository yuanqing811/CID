from data.img_utils import *
import cv2
from random import randint
from PIL import Image


def jpg_compression(array, quality):
    img = Image.fromarray(array)
    img.save('tmp.jpg', "JPEG", quality=quality)
    return cv2.cvtColor(cv2.imread('tmp.jpg'), cv2.COLOR_BGR2RGB)


def get_manipulated_image(image, c_manip=None):
    num_manipulations = 8
    if c_manip is None:
        c_manip = randint(0, num_manipulations-1)

    if c_manip == 0:
        return gamma_correction(image, 0.8), c_manip
    elif c_manip == 1:
        return gamma_correction(image, 1.2), c_manip
    elif c_manip == 2:
        return jpg_compression(image, 70), c_manip
    elif c_manip == 3:
        return jpg_compression(image, 90), c_manip
    elif c_manip == 4:
        return resizing(image, 0.5), c_manip
    elif c_manip == 5:
        return resizing(image, 0.8), c_manip
    elif c_manip == 6:
        return resizing(image, 1.5), c_manip
    elif c_manip == 7:
        return resizing(image, 2.0), c_manip
    else:
        raise ValueError('Unknown image manipulation type')
