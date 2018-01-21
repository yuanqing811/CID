import os
from data.datasets import Dataset
from keras import optimizers, losses, activations, models
from keras.layers import Dense, Input, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from data.dataset_utils import root_dir
import numpy as np
import pandas as pd
from data.dataset_utils import camera_names, cache_dir
from data.data_generators import DataGenerator

output_dir = os.path.join(root_dir, 'output')
model_data_dir = os.path.join(output_dir, 'model_data')
submission_dir = os.path.join(output_dir, 'submissions')

if not os.path.isdir(output_dir):
    print('Creating output directory')
    os.system('mkdir %s' % output_dir)

if not os.path.isdir(model_data_dir):
    print('Creating model data directory')
    os.system('mkdir %s' % model_data_dir)

if not os.path.isdir(submission_dir):
    print('Creating submissions directory')
    os.system('mkdir %s' % submission_dir)

num_cameras = len(camera_names)

data_set = Dataset('resnet')
batch_size = 25


def get_model():
    input_shape = (2048,)
    drop_prob = 0.2
    inp = Input(shape=input_shape)
    tmp = Dropout(rate=drop_prob)(inp)
    tmp = Dense(1024, activation=activations.relu)(tmp)
    tmp = Dropout(rate=drop_prob)(tmp)
    tmp = Dense(512, activation=activations.relu)(tmp)
    tmp = Dropout(rate=drop_prob)(tmp)
    tmp = Dense(256, activation=activations.relu)(tmp)
    tmp = Dropout(rate=drop_prob)(tmp)
    tmp = Dense(128, activation=activations.relu)(tmp)
    tmp = Dropout(rate=drop_prob)(tmp)
    tmp = Dense(64, activation=activations.relu)(tmp)
    tmp = Dropout(rate=drop_prob)(tmp)
    tmp = Dense(32, activation=activations.relu)(tmp)
    tmp = Dropout(rate=drop_prob)(tmp)
    dense_1 = Dense(num_cameras, activation=activations.softmax)(tmp)
    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam()
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model


def train_model(set_name):

    model_data_path = os.path.join(model_data_dir, set_name + "_xfer_resnet.best.hdf5")
    checkpoint = ModelCheckpoint(model_data_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=2)
    tb_callback = TensorBoard(log_dir='/tmp/cid/' + set_name, histogram_freq=0, batch_size=25, write_graph=True,
                              write_grads=False, write_images=False, embeddings_freq=0,
                              embeddings_layer_names=None, embeddings_metadata=None)

    callbacks_list = [checkpoint, early, tb_callback]

    data_gen = DataGenerator(hdf5_path=os.path.join(cache_dir, 'resnet.h5'),
                             subset_name=set_name, validation_split=0.9)
    n_train_batches = int((data_gen.n_train + batch_size - 1)/batch_size)
    n_valid_batches = int((data_gen.n_valid + batch_size - 1)/batch_size)

    model = get_model()

    train_data_generator = data_gen.generate(partition_name='train',
                                             keys=['x_res', 'y'],
                                             batch_size=batch_size)

    valid_data_generator = data_gen.generate(partition_name='valid',
                                             keys=['x_res', 'y'],
                                             batch_size=batch_size)

    model.fit_generator(generator=train_data_generator,
                        steps_per_epoch=n_train_batches,
                        epochs=25, verbose=2,
                        callbacks=callbacks_list,
                        validation_data=valid_data_generator,
                        validation_steps=n_valid_batches,
                        class_weight=None, max_queue_size=10, workers=1,
                        use_multiprocessing=False,
                        shuffle=False, initial_epoch=0)


def test_model(train_set_name, test_set_name, predictions_local):

    model_data_path = os.path.join(model_data_dir, train_set_name + "_xfer_resnet.best.hdf5")
    model = get_model()

    model.load_weights(filepath=model_data_path)

    data_gen = DataGenerator(hdf5_path=os.path.join(cache_dir, 'resnet.h5'),
                             subset_name=test_set_name)

    test_data_generator = data_gen.generate(keys=['x_res', 'image_index'], batch_size=1, for_keras=False)

    for x_tmp, image_index in test_data_generator:

        y_predicted = model.predict(x_tmp)

        if test_set_name == 'test_unalt':
            filename = data_set.test_unalt_set.map_index_to_filename(int(image_index[0]))
        else:  # 'test_manip'
            filename = data_set.test_manip_set.map_index_to_filename(int(image_index[0]))

        if filename in predictions_local.keys():
            # combine across multiple patches
            predictions_local[filename] += np.log(0.001 + y_predicted[0])
        else:
            predictions_local[filename] = np.log(0.001 + y_predicted[0])


if __name__ == '__main__':

    train_set_names = ['train_unalt', 'train_manip']
    test_set_names = ['test_unalt', 'test_manip']

    predictions = dict()
    for train_set_name, test_set_name in zip(train_set_names, test_set_names):
        train_model(train_set_name)
        test_model(train_set_name=train_set_name,
                   test_set_name=test_set_name,
                   predictions_local=predictions)

    test_files = list()
    final_predictions = list()

    for key, value in predictions.items():
        test_files.append(key)
        p_camera_index = np.argmax(value)
        final_predictions.append(camera_names[p_camera_index])

    df = pd.DataFrame(columns=['fname', 'camera'])
    df['fname'] = test_files
    df['camera'] = final_predictions
    sub_file = os.path.join(submission_dir, "test_resnet.csv")
    df.to_csv(sub_file, index=False)
