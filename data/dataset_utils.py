import numpy as np
import inspect
import os
import sys
from data.jpg_utils import get_img_data, crop, get_manipulated_image
from PIL import Image
import time

curr_filename = inspect.getfile(inspect.currentframe())
data_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
root_dir = data_dir.rsplit('/', 1)[0]
cache_name = 'data_cache'
cache_dir = os.path.join(data_dir, cache_name)

n_manip = 8

set_keys = {
    'train_unalt': ['x', 'x_coord', 'y', 'x_index'],
    'train_manip': ['x', 'x_coord', 'y', 'x_index', 'manip'],
    'test_unalt': ['x', 'x_coord', 'x_index'],
    'test_manip': ['x', 'x_coord', 'x_index']
}

camera_names = ['HTC-1-M7',
                'iPhone-6',
                'iPhone-4s',
                'LG-Nexus-5x',
                'Motorola-Droid-Maxx',
                'Motorola-Nexus-6',
                'Motorola-X',
                'Samsung-Galaxy-Note3',
                'Samsung-Galaxy-S4',
                'Sony-NEX-7'
                ]

camera_name_lookup = {'HTC-1-M7': 0,
                      'iPhone-6': 1,
                      'iPhone-4s': 2,
                      'LG-Nexus-5x': 3,
                      'Motorola-Droid-Maxx': 4,
                      'Motorola-Nexus-6': 5,
                      'Motorola-X': 6,
                      'Samsung-Galaxy-Note3': 7,
                      'Samsung-Galaxy-S4': 8,
                      'Sony-NEX-7': 9
                      }

abbr_camera_names = [
    'HTC-1-M7',
    'iP6',
    'iP4s',
    'LG5x',
    'MotoMax',
    'MotoNex6',
    'MotoX',
    'GalaxyN3',
    'GalaxyS4',
    'Nex7'
]

abbr_camera_name_lookup = {
    'HTC-1-M7': 0,
    'iP6': 1,
    'iP4s': 2,
    'LG5x': 3,
    'MotoMax': 4,
    'MotoNex6': 5,
    'MotoX': 6,
    'GalaxyN3': 7,
    'GalaxyS4': 8,
    'Nex7': 9
}

camera_dict = {
    'HTC-1-M7': 'HTC-1-M7',
    'iPhone-6': 'iP6',
    'iPhone-4s': 'iP4s',
    'LG-Nexus-5x': 'LG5x',
    'Motorola-Droid-Maxx': 'MotoMax',
    'Motorola-Nexus-6': 'MotoNex6',
    'Motorola-X': 'MotoX',
    'Samsung-Galaxy-Note3': 'GalaxyN3',
    'Samsung-Galaxy-S4': 'GalaxyS4',
    'Sony-NEX-7': 'Nex7'
}


def batch_fetch(x, batch_size, verbose=False):
    n_samples = x.shape[0]
    n_batches = int((n_samples + batch_size - 1)/batch_size)

    for i in range(n_batches):

        if verbose and i % 10 == 0 and i > 1:  # for display purpose only
            print('\rfetched data: {}/{}'.format(i, n_batches), end='', flush=True)
            sys.stdout.flush()

        batch_indices = slice(batch_size * i, min(batch_size * (i + 1), n_samples))
        yield x[batch_indices]
    print(' ... done')


class PatchGenerator(object):
    def __init__(self, patch_size, crop_size=(1024, 1024)):
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.patch_width = patch_size[0]
        self.patch_height = patch_size[1]

    def get_indices(self, img_size):
        try:
            (n_rows, n_cols) = img_size
        except:
            img_size = img_size.shape[:2]
            (n_rows, n_cols) = img_size

        offset = [0, 0]
        if self.crop_size:
            if img_size[0] > self.crop_size[0]:
                offset[0] = int((img_size[0] - self.crop_size[0]) / 2)
                n_rows = self.crop_size[0]
            if img_size[1] > self.crop_size[1]:
                offset[1] = int((img_size[1] - self.crop_size[1]) / 2)
                n_cols = self.crop_size[1]

        n_rows_patch = int((n_rows + self.patch_width - 1)/self.patch_width)
        n_cols_patch = int((n_cols + self.patch_height - 1)/self.patch_height)

        if n_rows_patch == 1:
            row_overlap = 0
        else:
            row_overlap = (self.patch_width * n_rows_patch - n_rows)/(n_rows_patch - 1)

        if n_cols_patch == 1:
            col_overlap = 0
        else:
            col_overlap = (self.patch_height * n_cols_patch - n_cols)/(n_cols_patch - 1)

        if row_overlap > 0:
            rows = np.arange(offset[0], n_rows - row_overlap - 1 + offset[0], self.patch_width - row_overlap).astype(np.uint16)
        else:
            rows = np.arange(offset[0], n_rows + offset[0], self.patch_width).astype(np.uint16)

        if col_overlap > 0:
            cols = np.arange(offset[1], n_cols - col_overlap - 1 + offset[1], self.patch_height - col_overlap).astype(np.uint16)
        else:
            cols = np.arange(offset[1], n_cols + offset[1], self.patch_height).astype(np.uint16)

        indices = np.empty((n_rows_patch * n_cols_patch, 2), dtype=np.int16)
        k = 0
        for row in rows:
            for col in cols:
                indices[k][0] = row
                indices[k][1] = col
                k += 1
        return indices

    def extract_patches(self, image, include_indices=False):
        if len(image.shape) < 3:
            image = image[:, :, None]

        n_channels = image.shape[2]
        indices = self.get_indices(image)

        n_patches = indices.shape[0]
        data_shape = (n_patches, self.patch_width, self.patch_height, n_channels)
        patches = np.empty(data_shape, dtype=np.uint8)
        for i, (row, col) in enumerate(indices):
            try:
                patches[i] = image[row:row+self.patch_width, col:col+self.patch_height, :]
            except ValueError:
                print(row, col, image.shape[0], image.shape[1])
                raise

        return patches, indices if include_indices else patches


train_dir = os.path.join(data_dir, 'train')
train_manip_dir = os.path.join(data_dir, 'train_manip')


def create_data(self):
    # check to see if manip_train_directory is there
    if not os.path.isdir(train_manip_dir):
        print('Creating train_manip directory')
        os.system('mkdir %s' % train_manip_dir)

    for camera_name in camera_names:
        print('Processing camera: ', camera_name)
        camera_dir = os.path.join(train_manip_dir, camera_name)

        if not os.path.isdir(camera_dir):
            os.system("mkdir %s" % camera_dir)

        for index, filename in enumerate(self.train_set.get_camera_filenames(camera_name)):
            if index % 20 == 0:
                print('\rFinished processing this many images: ', index, end='')
            create_image(camera_name, filename)


def create_image(camera_name, filename):
    img_path = os.path.join(train_dir, camera_name, filename)
    with Image.open(img_path) as img_file:
        img_data = get_img_data(img_file)

    for c_manip in range(n_manip):
        img, _ = get_manipulated_image(img_data, c_manip)

        filename = filename.rsplit('.', 1).lower()
        new_filename = filename + '_manip' + str(c_manip) + '.tif'

        new_img_path = os.path.join(train_manip_dir, camera_name, new_filename)

        im = Image.fromarray(img)
        im = crop(im)
        im._save(new_img_path)


def seconds_to_hhmmss(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def print_time_estimate(old_time, current_image, total_images):
    new_time = time.time()
    elapsed_time = new_time - old_time
    time_per_image = elapsed_time / (current_image + 1)
    remain_time = time_per_image * (total_images - current_image)

    print('\rprocessed {}/{} files, '
          'elapsed time: {}, '
          'estimated remaining time: {}, '.format(current_image + 1, total_images,
                                                  seconds_to_hhmmss(elapsed_time),
                                                  seconds_to_hhmmss(remain_time), ),
          end='', flush=True)
    sys.stdout.flush()

