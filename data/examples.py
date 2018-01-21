from data.datasets import Dataset
from data.jpg_utils import get_img_data
from PIL import Image
from data.data_generators import DataGenerator
from data.dataset_utils import cache_dir
import os


def get_rgb_data(img_file_name):
    with Image.open(img_file_name) as img_file:
        img_data = get_img_data(img_file,  'RGB')
    return img_data


def create_resnet_dataset(h5_file='resnet'):
    dataset = Dataset(h5_name=h5_file)
    dataset.save_dataset(dataset_name='rgb_224',
                         data_shape=(224, 224, 3),
                         data_func=get_rgb_data,
                         crop_size={
                     'train_unalt': (1120, 1120),
                     'train_manip': (448, 448),
                     'test_unalt': (448, 448),
                     'test_manip': (448, 448)
                 })


def compute_resnet_feature(h5_file='resnet'):
    from data.transfer_utils import get_resnet50_features
    dataset = Dataset(h5_name=h5_file)
    dataset.save_feature(dataset_name='rgb_224',
                         feature_name='x_res',
                         data_shape=(2048, ),
                         data_func=get_resnet50_features)


def load_shuffled_dataset(h5_file='resnet'):
    data_gen = DataGenerator(hdf5_path=os.path.join(cache_dir, h5_file + '.h5'),
                             subset_name='train_unalt', validation_split=0.9)
    for x, y in data_gen.generate(partition_name='train',
                                  keys=['x', 'y'],
                                  batch_size=10):
        print(x.shape)
        print(y.shape)
        break


if __name__ == '__main__':
    file_name = 'resnet_new'
#    create_resnet_dataset(file_name)
#    compute_resnet_feature(file_name)
    load_shuffled_dataset(file_name)

