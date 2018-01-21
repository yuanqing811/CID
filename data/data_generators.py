import tables
from data.dataset_utils import set_keys
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

        self.train_samples = list()
        self.valid_samples = list()

        with tables.open_file(self.hdf5_path, mode='r') as hdf5_file:
            if self.dataset_name not in hdf5_file.root:
                raise Exception("dataset with name '%s' doesn't exist" % self.dataset_name)
            dataset_group = hdf5_file.get_node(hdf5_file.root, self.dataset_name)
            set_group = hdf5_file.get_node(dataset_group, subset_name)
            self.n_samples = int(set_group.x.shape[0])

        if 'train' in subset_name:
            shuffled_samples = np.arange(self.n_samples)
            np.random.shuffle(shuffled_samples)
            num_training_samples = int(self.n_samples * validation_split)
            self.train_samples = shuffled_samples[0:num_training_samples]
            self.valid_samples = shuffled_samples[num_training_samples:self.n_samples]

    def generate(self, partition_name=None, keys=None, batch_size=10, for_keras=True):
        if keys is None:
            keys = set_keys[self.subset_name]

        data_types = {'x':'uint8', 'y':'uint8', 'x_res':'float', 'patch_coord':'int', 'image_index':
                      'int'}

        # open the hdf5 file
        with tables.open_file(self.hdf5_path, mode='r') as hdf5_file:
            set_group = hdf5_file.get_node(hdf5_file.root, '/%s/%s/' % (self.dataset_name, self.subset_name))
            data = [(key, hdf5_file.get_node(set_group, key)) for key in keys]
            if 'train' in self.subset_name and for_keras is True:
                """Generates batches of samples"""
                # Infinite loop
                while 1:
                    # Create a random shuffle
                    if partition_name == 'train':
                        n_samples = len(self.train_samples)
                        np.random.shuffle(self.train_samples)
                        shuffled_indices = self.train_samples
                    elif partition_name == 'valid':
                        n_samples = len(self.valid_samples)
                        np.random.shuffle(self.valid_samples)
                        shuffled_indices = self.valid_samples
                    else:
                        raise ValueError('Unknown partition for training set!')
                    # Generate batches
                    shuffled_indices = np.asarray(shuffled_indices)

                    for i_s in range(0, n_samples, batch_size):
                        batch_indices = shuffled_indices[i_s:min(i_s + batch_size, n_samples)]
                        output = list()

                        # creating buffer arrays
                        for key, x in data:
                            output_tmp = np.zeros(shape=(batch_indices.shape[0], ) + x.shape[1:], dtype=data_types[key])
                            for i_b in range(batch_indices.shape[0]):
                                output_tmp[i_b] = x[batch_indices[i_b]]
                            output.append(encode_predictions(output_tmp, self.n_classes) if key == 'y' else output_tmp)

                        yield output
            else:
                n_samples = data[0][1].shape[0]
                # Generate batches
                for i_s in range(0, n_samples, batch_size):
                    batch_indices = slice(i_s, i_s + batch_size)
                    output = [encode_predictions(x[batch_indices], self.n_classes) if key == 'y'
                              else x[batch_indices] for key, x in data]
                    yield output


def decode_predictions(y_sparse):
    return np.argmax(y_sparse, axis=1)


def encode_predictions(y, n_classes):
    y_sparse = np.zeros((y.shape[0], n_classes))
    y_sparse[np.arange(y.shape[0]), y] = 1
    return y_sparse
