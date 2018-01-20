from data.datasets import Dataset
from data.jpg_utils import get_img_data
from PIL import Image


def get_rgb_data(filename):
    with Image.open(filename) as img_file:
        img_data = get_img_data(img_file,  'RGB')
    return img_data


def create_resnet_dataset():
    dataset = Dataset(h5_name='resnet')
    dataset.save_dataset(dataset_name='rgb_224',
                         data_shape=(224, 224, 3),
                         data_func=get_rgb_data,
                         crop_size={
                     'train_unalt': (1120, 1120),
                     'train_manip': (448, 448),
                     'test_unalt': (448, 448),
                     'test_manip': (448, 448)
                 })


def compute_resnet_feature():
    from data.transfer_utils import get_resnet50_features

    dataset = Dataset(h5_name='resnet')
    dataset.save_feature(dataset_name='rgb_224',
                         feature_name='x_res',
                         data_shape=(2048, ),
                         data_func=get_resnet50_features)


def load_shuffled_dataset():
    dataset = Dataset(h5_name='resnet')
    data_gen = dataset.load_dataset('rgb_224')
    for x, y in data_gen.generate(set_name='train_manip',
                                  partition_name='train',
                                  keys=['x', 'y'],
                                  batch_size=10):
        print(x.shape)
        print(y.shape)
        print(y)
        break


if __name__ == '__main__':
    # create_resnet_dataset()
    # compute_resnet_feature()
    # shuffle_resnet_dataset()
    load_shuffled_dataset()

