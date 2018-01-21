import numpy as np
import os
import sys
from data.dataset_utils import batch_fetch, PatchGenerator, seconds_to_hhmmss
from data.dataset_utils import cache_dir
from data.ImageDataset import TrainingSet, TestSet
import tables
import time
from data.dataset_utils import print_time_estimate

set_names = ['train_unalt', 'test_unalt', 'train_manip', 'test_manip']

data_src_dir = ['train_unalt',
                'test_unalt',
                'train_manip',
                'test_manip']


class Dataset(object):
    def __init__(self, h5_name):
        self.hdf5_name = '%s.h5' % h5_name
        self.hdf5_path = os.path.join(cache_dir, self.hdf5_name)
        self.hdf5_write_mode = 'a'  # default to append mode, can be changed to 'w' mode using overwrite

        self.train_unalt_set = TrainingSet(manip=False)
        self.train_manip_set = TrainingSet(manip=True)
        self.test_unalt_set = TestSet(manip=False)
        self.test_manip_set = TestSet(manip=True)

        self.data_src = {
            'train_unalt': self.train_unalt_set,
            'train_manip': self.train_manip_set,
            'test_unalt': self.test_unalt_set,
            'test_manip': self.test_manip_set
        }

        print('# train images: %d\n'
              '# train_manip images: %d\n'
              '# test images: %d\n'
              '# test_manip images: %d\n' % (self.train_unalt_set.n_files,
                                             self.train_manip_set.n_files,
                                             self.test_unalt_set.n_files,
                                             self.test_manip_set.n_files))

        self.feature_dtype = tables.Float32Atom()

        self.data_type = {
            'x': tables.UInt8Atom(),
            'y': tables.UInt8Atom(),
            'image_index': tables.UInt16Atom(),
            'patch_coord': tables.UInt16Atom(),
            'manip': tables.UInt8Atom(),
        }

        self.data_shape = {
            'y': (0,),
            'patch_coord': (0, 2),
            'image_index': (0, ),
            'manip': (0, )
        }

        self.filters = tables.Filters(complevel=5, complib='blosc')
        self.label_shape = (0,)
        self.coord_shape = (0, 2)
        self.index_shape = (0, )
        self.manip_shape = (0, )

    def save_dataset(self, dataset_name, data_shape, data_func, crop_size=None, max_n_images=None, write=False):
        # initialization
        if write is True or write is False:
            write = dict([(set_name, write) for set_name in set_names])

        if crop_size is None or type(crop_size) == tuple:
            write = dict([(set_name, crop_size) for set_name in set_names])

        patch_size = data_shape[:2]

        hdf5_file = tables.open_file(self.hdf5_path, mode='a')

        if dataset_name not in hdf5_file.root:
            dataset_group = hdf5_file.create_group(hdf5_file.root, dataset_name)
        else:
            dataset_group = hdf5_file.get_node(hdf5_file.root, dataset_name)

        for set_name in set_names:
            set_write = write.get(set_name, False)
            if set_name not in dataset_group:
                set_write = True

            if set_write is False:
                continue

            print('creating set %s ...' % set_name)

            if set_name in dataset_group:  # set_overwrite is True
                hdf5_file.remove_node(dataset_group, set_name, recursive=True)

            set_group = hdf5_file.create_group(dataset_group, set_name)

            patch_generator = PatchGenerator(patch_size=patch_size, crop_size=crop_size[set_name])

            set_crop_size = crop_size[set_name]
            n_rows = int((set_crop_size[0] + patch_size[0] - 1)/patch_size[0])
            n_cols = int((set_crop_size[1] + patch_size[1] - 1)/patch_size[1])
            n_patches_per_image = n_rows * n_cols

            n_files = self.data_src[set_name].n_files
            n_samples = n_files * n_patches_per_image
            data_generator = self.data_src[set_name].generate(patch_generator, data_func, max_n_images)
            self._save(hdf5_file, set_group, (n_samples,) + data_shape,
                       self.data_src[set_name].keys, data_generator)

        hdf5_file.close()

    def _save(self, hdf5_file, group, data_shape, keys, data_generator):
        n_samples = data_shape[0]
        self.data_shape['x'] = (0, ) + data_shape[1:]

        nodes = []
        for key in keys:
            nodes.append(
                hdf5_file.create_earray(group, key,
                                        atom=self.data_type[key],
                                        shape=self.data_shape[key],
                                        filters=self.filters,
                                        expectedrows=n_samples)
            )

        # todo: allow data_gen to include_indices, verbose option, and check order of data
        for data in data_generator:
            n_patches = data[0].shape[0]  # hack: assuming all data has field x
            for i in range(len(keys)):
                key = keys[i]
                if key in ['y', 'image_index', 'manip']:
                    nodes[i].append(np.array([data[i], ] * n_patches, dtype=self.data_type[key]))
                else:   # x, patch_coord
                    nodes[i].append(data[i])

    def save_feature(self, dataset_name, feature_name, data_shape, data_func, batch_size=10, write=False):
        if write is True or False:
            write = dict([(h5_path.split('/', maxsplit=1)[0], write) for h5_path in data_src_dir])

        hdf5_file = tables.open_file(self.hdf5_path, mode='a')
        if dataset_name not in hdf5_file.root:
            raise Exception("dataset with name '%s' doesn't exist" % dataset_name)

        dataset = hdf5_file.get_node(hdf5_file.root, dataset_name)

        for h5_path in data_src_dir:
            set_name = h5_path.split('/', maxsplit=1)[0]
            print('processing feature %s for subset %s @ %s' % (feature_name, set_name, h5_path))
            subset = hdf5_file.get_node(dataset, h5_path)
            x_src = subset.x
            n_samples, n_rows, n_cols, n_channels = x_src.shape

            if feature_name in subset:
                set_name = h5_path.split('/', maxsplit=1)[0]

                if write[set_name] is True:
                    hdf5_file.remove_node(subset, feature_name)
                else:   # do not overwrite
                    continue

            x_dst = hdf5_file.create_earray(subset, feature_name,
                                            self.feature_dtype,
                                            shape=(0,) + data_shape,
                                            expectedrows=n_samples)

            old_time = time.time()
            for i, x_batch in enumerate(batch_fetch(x_src, batch_size=batch_size, verbose=False)):
                x_dst.append(data_func(x_batch))
                print_time_estimate(old_time, i * batch_size, n_samples)

        print('...done')
        hdf5_file.close()

