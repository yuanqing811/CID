from data.img_utils import *
import cv2
from random import randint

samplings = {(1, 1, 1, 1, 1, 1): 0,
             (2, 1, 1, 1, 1, 1): 1,
             (2, 2, 1, 1, 1, 1): 2,
             }


def zigzag(n):
    # JpegImagePlugin.convert_dict_qtables()
    order = sorted(((x, y) for x in range(n) for y in range(n)),
                   key=lambda x: (x[0]+x[1], -x[1] if (x[0]+x[1]) % 2 else x[1]))
    return order


def qt_arr2mat(arr):
    mat = np.zeros((8, 8), dtype=np.int16)

    for idx, (x, y) in enumerate(zigzag(8)):
        mat[x][y] = arr[idx]
    return mat


def get_exif(img):
    exif = dict((ExifTags.TAGS[k], v) for k, v in img._getexif().items() if k in ExifTags.TAGS)
    return exif


def get_jpg_info(img, **options):

    """
    Chroma Subsampling:

    Subsampling is the practice of encoding images by implementing less resolution for chroma information
    than for luma information. (ref.: https://en.wikipedia.org/wiki/Chroma_subsampling)

    Possible subsampling values are 0, 1 and 2 that correspond to 4:4:4, 4:2:2 and 4:1:1 (or 4:2:0?).
    4:1:1 - 1/4 horizontal resolution, full vertical resolution
    4:2:2 = 1/2 horizontal resolution, full vertical resolution
    4:4:4 - full horizontal resolution, full vertical resolution
    You can get the chroma subsampling of a JPEG with the JpegImagePlugin.get_subsampling(im) function.

    Orientation: (letter F)

      1        2       3      4         5            6           7          8

    888888  888888      88  88      8888888888  88                  88  8888888888
    88          88      88  88      88  88      88  88          88  88      88  88
    8888      8888    8888  8888    88          8888888888  8888888888          88
    88          88      88  88
    88          88  888888  888888

    """

    output = {}
    if options.get('qt', False) or options.get('quantization', False):
        output['qt'] = img.quantization[0]
    if options.get('subsampling', False):
        output['subsampling'] = JpegImagePlugin.get_sampling(img)
    if options.get('orientation', False):
        exif = get_exif(img)
        output['orientation'] = exif.get('Orientation', None)
    return output


def get_img_alignment(img_data):
    patch_size = 8
    image_x = 16    #int(B.shape[0]/8)
    image_y = 16    #int(B.shape[1]/8)

    offset_metric = np.zeros((patch_size, patch_size))

    for i_x in range(image_x):
        for i_y in range(image_y):
            for i_offset_x in range(patch_size):
                for i_offset_y in range(patch_size):
                    patch = img_data[patch_size * i_x + i_offset_x:patch_size * i_x + patch_size + i_offset_x,
                            patch_size * i_y + i_offset_y:patch_size * i_y + patch_size + i_offset_y]

                    patch_dct = dct2(patch)
                    patch_dct = np.int16(patch_dct)

                    offset_metric[i_offset_x][i_offset_y] = offset_metric[i_offset_x][i_offset_y] + np.count_nonzero(patch_dct)

    min_index = np.unravel_index(offset_metric.argmin(), offset_metric.shape)

#    print(offset_metric, min_index)

    return min_index


def jpg_compression(array, quality):
    img = Image.fromarray(array)
    img._save('img.jpg', "JPEG", quality=quality)
    return cv2.cvtColor(cv2.imread('img.jpg'), cv2.COLOR_BGR2RGB)


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
