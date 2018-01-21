import os
from data.jpg_utils import get_img_data, crop, get_manipulated_image
from data.dataset_utils import data_dir
from data.dataset_utils import camera_names
from PIL import Image
from data.ImageDataset import TrainingSet

train_dir = os.path.join(data_dir, 'train')
train_manip_dir = os.path.join(data_dir, 'train_manip')


def create_data(t_set):
    # check to see if manip_train_directory is there
    if not os.path.isdir(train_manip_dir):
        print('Creating train_manip directory')
        os.system('mkdir %s' % train_manip_dir)

    for camera_name in camera_names:
        print('Processing camera: ', camera_name)
        camera_dir = os.path.join(train_manip_dir, camera_name)

        if not os.path.isdir(camera_dir):
            os.system("mkdir %s" % camera_dir)

        for index, filename in enumerate(t_set.get_camera_filenames(camera_name)):
            if index % 20 == 0:
                print('\rFinished processing this many images: %d ' % index, end='')
            create_image(camera_name, filename)


def create_image(camera_name, filename):
    img_path = os.path.join(train_dir, camera_name, filename)
    with Image.open(img_path) as img_file:
        img_data = get_img_data(img_file)

    n_manip = 8
    for c_manip in range(n_manip):
        img, _ = get_manipulated_image(img_data, c_manip)

        filename = filename.rsplit('.')[0]
        new_filename = filename + '_manip' + str(c_manip) + '.tif'

        new_img_path = os.path.join(train_manip_dir, camera_name, new_filename)

        im = Image.fromarray(img)
        im = crop(im)
        im.save(new_img_path)


if __name__ == '__main__':
    training_set = TrainingSet(manip=False)
    create_data(training_set)

