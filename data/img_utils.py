import numpy as np
from scipy.fftpack import dct
import cv2


def dct2(a):
    return dct(dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def get_img_data(img_file, mode=None, data_format=None):
    """
    imread uses the Python Imaging Library (PIL) to read an image.
    The following notes are from the PIL documentation.

    mode can be one of the following strings:

    ‘L’ (8-bit pixels, black and white)
    ‘P’ (8-bit pixels, mapped to any other mode using a color palette)
    ‘RGB’ (3x8-bit pixels, true color)
    ‘RGBA’ (4x8-bit pixels, true color with transparency mask)
    ‘CMYK’ (4x8-bit pixels, color separation)
    ‘YCbCr’ (3x8-bit pixels, color video format)
    ‘I’ (32-bit signed integer pixels)
    ‘F’ (32-bit floating point pixels)
    """
    if mode:
        img_data = np.asarray(img_file.convert(mode), dtype=np.uint8)
    else:
        img_data = np.asarray(img_file, dtype=np.uint8)
    return img_data


def extract_2d_patches(image_data, patch_size, offset=(0, 0)):
    for row in range(offset[0], image_data.shape[0], patch_size):
        for col in range(offset[1], image_data.shape[1], patch_size):
            patch = image_data[row:row+patch_size, col:col+patch_size]
            yield patch


def crop(img, crop_size=512):
    if type(crop_size) in [tuple, list]:
        crop_width, crop_height = crop_size
    else:
        crop_width = crop_size
        crop_height = crop_size

    width, height = img.size  # Get dimensions

    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = (width + crop_width) / 2
    bottom = (height + crop_height) / 2

    return img.crop((left, top, right, bottom))


def gamma_correction(array_img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(array_img, table)


def resizing(array_img, factor):
    h, w, ch = array_img.shape
    return cv2.resize(array_img, (int(factor * w), int(factor * h)), interpolation=cv2.INTER_CUBIC)