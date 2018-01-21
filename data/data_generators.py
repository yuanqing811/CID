import tables
from data.dataset_utils import set_keys, numpy_dtype
import numpy as np


class DataGenerator(object):
    """
    After adding data and lossless-ly compressing using EArray,
    we can also slice subsets of the EArray (e.g. hdf5_file.root.data[10:100])
    which only brings that portion of the data into memory.
    """

    def __init__(self, hdf5_path, subset_name, dataset_name='rgb_224', validation_split=0.9):
        self.hdf5_path = hdf5_path
        self.dataset_name = dataset_name    # ex. rgb224
        self.n_classes = 10              # 10 (used for sparsify only)
        self.n_samples = 0
        self.subset_name = subset_name

        with tables.open_file(self.hdf5_path, mode='r') as hdf5_file:
            if self.dataset_name not in hdf5_file.root:
                raise Exception("dataset with name '%s' doesn't exist" % self.dataset_name)
            dataset_group = hdf5_file.get_node(hdf5_file.root, self.dataset_name)
            set_group = hdf5_file.get_node(dataset_group, subset_name)
            self.n_samples = int(set_group.x.shape[0])

        if 'train' in subset_name:
            self.partitions = {}
            indices = np.random.permutation(self.n_samples)
            self.n_train = int(self.n_samples * validation_split)
            self.n_valid = self.n_samples - self.n_train
            self.partitions['train'] = indices[:self.n_train]
            self.partitions['valid'] = indices[self.n_train:]

    def generate(self, partition_name=None, keys=None, batch_size=10, for_keras=True):
        if keys is None:
            keys = set_keys[self.subset_name]

        # open the hdf5 file
        with tables.open_file(self.hdf5_path, mode='r') as hdf5_file:
            set_group = hdf5_file.get_node(hdf5_file.root,
                                           '/%s/%s/' % (self.dataset_name, self.subset_name))
            nodes = [hdf5_file.get_node(set_group, key) for key in keys]
            buffers = [np.empty(shape=(batch_size, ) + nodes[i].shape[1:],
                                dtype=numpy_dtype[keys[i]]) for i in range(len(keys))]

            if 'train' in self.subset_name and for_keras is True:
                indices = self.partitions[partition_name]
                n_samples = indices.shape[0]

                """Generates batches of samples"""
                # Infinite loop
                while 1:
                    np.random.shuffle(indices)

                    # Generate batches
                    for i_s in range(0, n_samples, batch_size):
                        batch_indices = indices[i_s:i_s + batch_size]
                        n_batch = batch_indices.shape[0]
                        for i in range(len(keys)):
                            node = nodes[i]
                            for j in range(n_batch):
                                buffers[j] = node[batch_indices[j]]

                        yield [encode_predictions(buffers[i][:n_batch], self.n_classes)
                               if keys[i] == 'y' else buffers[i][:n_batch]
                               for i in range(len(keys))]
            else:
                n_samples = nodes[0][1].shape[0]

                # Generate batches
                for i_s in range(0, n_samples, batch_size):
                    batch_indices = slice(i_s, i_s + batch_size)
                    yield [encode_predictions(nodes[i][batch_indices], self.n_classes)
                           if keys[i] == 'y' else nodes[i][batch_indices]
                           for i in range(len(keys))]


def decode_predictions(y_sparse):
    return np.argmax(y_sparse, axis=1)


def encode_predictions(y, n_classes):
    y_sparse = np.zeros((y.shape[0], n_classes))
    y_sparse[np.arange(y.shape[0]), y] = 1
    return y_sparse
