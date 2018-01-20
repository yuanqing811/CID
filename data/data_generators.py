import tables
from data.dataset_utils import set_keys
import numpy as np

class DataGenerator(object):
    """
    After adding data and lossless-ly compressing using EArray,
    we can also slice subsets of the EArray (e.g. hdf5_file.root.data[10:100])
    which only brings that portion of the data into memory.
    """

    def __init__(self, hdf5_path, dataset_name, n_classes):
        self.hdf5_path = hdf5_path
        self.dataset_name = dataset_name    # ex. rgb224
        self.n_classes = n_classes              # 10 (used for sparsify only)
        self.n_samples = {
            'train_unalt': {
                'train': 0,
                'valid': 0,
            },
            'train_manip': {
                'train': 0,
                'valid': 0,
            },
            'test_unalt': 0,
            'test_manip': 0,
        }

        with tables.open_file(self.hdf5_path, mode='r') as hdf5_file:
            if self.dataset_name not in hdf5_file.root:
                raise Exception("dataset with name '%s' doesn't exist" % self.dataset_name)
            dataset_group = hdf5_file.get_node(hdf5_file.root, self.dataset_name)

            for set_name in self.n_samples:
                set_group = hdf5_file.get_node(dataset_group, set_name)
                if 'test' in set_name:
                    n_samples = set_group.x.shape[0]
                    self.n_samples[set_name] = int(n_samples)
                else:
                    for partition_name in ['train', 'valid']:
                        if partition_name == 'train':
                            n_samples = set_group.train.x.shape[0]
                        elif partition_name == 'valid':
                            n_samples = set_group.valid.x.shape[0]
                        else:
                            raise Exception('unknown partition %s' % partition_name)
                        self.n_samples[set_name][partition_name] = int(n_samples)

    def generate(self, set_name, partition_name=None, keys=None, batch_size=10, for_keras=True):
        if keys is None:
            keys = set_keys[set_name]

        # open the hdf5 file
        with tables.open_file(self.hdf5_path, mode='r') as hdf5_file:
            set_group = hdf5_file.get_node(hdf5_file.root, '/%s/%s/' % (self.dataset_name, set_name))

            if 'test' in set_name:
                if partition_name:
                    raise Exception('no partitioning for test')
                data = [(key, hdf5_file.get_node(set_group, key)) for key in keys]
            else:
                if partition_name == 'train':
                    try:
                        data = [(key, hdf5_file.get_node(set_group.train, key)) for key in keys]
                    except:
                        print(hdf5_file.list_nodes(set_group))
                        raise
                elif partition_name == 'valid':   # partition_name == 'valid'
                    try:
                        data = [(key, hdf5_file.get_node(set_group.valid, key)) for key in keys]
                    except:
                        print(hdf5_file.list_nodes(set_group))
                        raise
                else:
                    data = [(key, hdf5_file.get_node(set_group.train, key)) for key in keys]
            n_samples = data[0][1].shape[0]

            if 'train' in set_name and for_keras is True:
                """Generates batches of samples"""
                # Infinite loop
                while 1:
                    # Generate batches
                    for i_s in range(0, n_samples, batch_size):
                        batch_indices = slice(i_s, i_s + batch_size)
                        output = [encode_predictions(x[batch_indices], self.n_classes) if key == 'y' else x[batch_indices] for key, x in data]
                        yield output
            else:
                # Generate batches
                for i_s in range(0, n_samples, batch_size):
                    batch_indices = slice(i_s, i_s + batch_size)
                    output = [encode_predictions(x[batch_indices], self.n_classes) if key == 'y' else x[batch_indices] for key, x in data]
                    yield output


def decode_predictions(y_sparse):
    return np.argmax(y_sparse, axis=1)


def encode_predictions(y, n_classes):
    y_sparse = np.zeros((y.shape[0], n_classes))
    y_sparse[np.arange(y.shape[0]), y] = 1
    return y_sparse