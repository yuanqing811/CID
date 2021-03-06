import os
import re
import time
import numpy as np
from data.dataset_utils import abbr_camera_name_lookup, camera_name_lookup, camera_names, abbr_camera_names, data_dir
from data.dataset_utils import print_time_estimate


class ImageDataset(object):
    def __init__(self, name,
                 directory_filter_by=None, filename_filter_by=None,
                 directory_sort_by=None, filename_sort_by=None):

        self.name = name
        self.data_dir = os.path.join(data_dir, name)
        self.filenames = []
        self.directories = []
        self.n_files = 0
        self.n_directories = 0
        self.directory_filenames = {}

        self.directory_filter_by = directory_filter_by
        self.filename_filter_by = filename_filter_by
        self.directory_sort_by = directory_sort_by
        self.filename_sort_by = filename_sort_by

        self.keys = ['x', 'patch_coord', 'image_index']

        self.load()

    def load(self):
        if not os.path.isdir(self.data_dir):
            raise Exception('directory for %s doesn\'t exist' % type(self))

        for root, directories, filenames in os.walk(self.data_dir):
            filenames = list(filter(lambda x: not x.startswith('.'), filenames))
            if self.filename_filter_by:
                filenames = list(filter(self.filename_filter_by, filenames))
            if self.filename_sort_by:
                filenames.sort(key=self.filename_sort_by)
            else:
                filenames.sort()
            self.filenames.extend(filenames)

            directories = list(filter(lambda x: os.path.isdir(os.path.join(self.data_dir, x)), directories))
            if self.directory_filter_by:
                directories = list(filter(self.directory_filter_by, directories))
            if self.directory_sort_by:
                directories.sort(key=self.directory_sort_by)
            else:
                directories.sort()
            self.directories.extend(directories)

            for directory in self.directories:
                filenames = os.listdir(os.path.join(self.data_dir, directory))
                filenames = list(filter(lambda x: not x.startswith('.'), filenames))
                if self.filename_filter_by:
                    filenames = list(filter(self.filename_filter_by, filenames))
                if self.filename_sort_by:
                    filenames.sort(key=self.filename_sort_by)
                else:
                    filenames.sort()

                self.directory_filenames[directory] = filenames
                self.filenames.extend([os.path.join(directory, filename) for filename in filenames])

            break

        self.n_directories = len(self.directories)
        self.n_files = len(self.filenames)

    def load_directory_filenames(self, directory, max_n_images=None):
        if max_n_images is None:
            max_n_images = self.n_files
        return self.directory_filenames[directory][:max_n_images]

    def generate(self, patch_generator, data_func, max_n_images=None, verbose=True):
        if max_n_images is None:
            max_n_images = self.n_files

        old_time = time.time()
        for i in range(max_n_images):
            filename = self.filenames[i]

            x = data_func(os.path.join(self.data_dir, filename))
            x, patch_coord = patch_generator.extract_patches(x, include_indices=True)
            n_patches = x.shape[0]
            file_idx = np.array([i, ] * n_patches, dtype=np.uint16)

            yield x, patch_coord, file_idx

            if verbose and (i + 1) % 10 == 0:  # for display purpose only
                print_time_estimate(old_time, i, max_n_images)

        print('...done')

    def map_index_to_filename(self, idx, abs_path=False):
        if abs_path:
            return os.path.join(self.data_dir, self.filenames[idx])
        else:
            return self.filenames[idx]


class TrainingSet(ImageDataset):
    def __init__(self, manip):

        self.manip = manip

        super(TrainingSet, self).__init__('train_manip' if manip else 'train',
                                          directory_sort_by=type(self).get_directory_indices,
                                          filename_sort_by=self.get_file_indices)

        self.n_classes = len(camera_names)
        self.keys = ['x', 'patch_coord', 'y', 'image_index']
        if manip:
            self.keys.append('manip')

    @classmethod
    def get_directory_indices(cls, directory):
        return camera_name_lookup[directory]

    def get_file_indices(self, filename):
        if self.manip:
            try:
                abbr_camera_name, file_idx, manip_idx = re.findall(r'\(([A-Za-z0-9\-]+)\)(\d+)_manip(\d+)\.tif',
                                                                   filename, re.IGNORECASE)[0]
            except:
                print(filename)
                raise

            return abbr_camera_name_lookup[abbr_camera_name], int(file_idx), int(manip_idx)
        else:
            try:
                abbr_camera_name, file_idx = re.findall(r'\(([A-Za-z0-9\-]+)\)(\d+)\.jpe?g',
                                                        filename, re.IGNORECASE)[0]
            except:
                print(filename)
                raise

            return abbr_camera_name_lookup[abbr_camera_name], int(file_idx)

    def get_camera_filenames(self, camera_name, max_num_images=None):
        filenames = self.load_directory_filenames(directory=camera_name, max_n_images=max_num_images)
        return filenames

    def map_index_to_filename(self, indices, abs_path=False):
        if self.manip:
            camera_idx, file_idx, manip_idx = indices
            filename = '(%s)%d_manip%d.tif' % (abbr_camera_names[camera_idx], file_idx, manip_idx)
        else:
            camera_idx, file_idx = indices
            filename = '(%s)%d.jpg' % (abbr_camera_names[camera_idx], file_idx)

        if abs_path:
            camera_name = camera_names[camera_idx]
            return os.path.join(self.data_dir, camera_name, filename)
        else:
            return filename

    def generate(self, patch_generator, data_func, max_n_images=None, verbose=True):
        if max_n_images is None or max_n_images > self.n_files:
            max_n_images = self.n_files

        old_time = time.time()
        for i in range(max_n_images):
            filename = self.filenames[i]

            x = data_func(os.path.join(self.data_dir, filename))
            x, patch_coord = patch_generator.extract_patches(x, include_indices=True)
            n_patches = x.shape[0]

            if self.manip:
                y, file_idx, manip_idx = self.get_file_indices(filename)
                y = np.array([y, ] * n_patches, dtype=np.uint8)
                file_idx = np.array([file_idx, ] * n_patches, dtype=np.uint16)
                manip_idx = np.array([manip_idx, ] * n_patches, dtype=np.uint8)
                yield x, patch_coord, y, file_idx, manip_idx
            else:
                y, file_idx = self.get_file_indices(filename)
                y = np.array([y, ] * n_patches, dtype=np.uint8)
                file_idx = np.array([file_idx, ] * n_patches, dtype=np.uint16)
                yield x, patch_coord, y, file_idx

            if verbose and (i + 1) % 10 == 0:  # for display purpose only
                print_time_estimate(old_time, i, max_n_images)

        print('... done')


class TestSet(ImageDataset):
    def __init__(self, manip=False):
        test_dir = os.path.join(data_dir, 'test')
        self.manip = manip

        super(TestSet, self).__init__(test_dir, filename_filter_by=self.filename_filter_func)

    def filename_filter_func(self, filename):
        if self.manip:
            return 'manip' in filename
        else:
            return 'manip' not in filename
