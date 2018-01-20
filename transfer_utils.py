import keras.backend as K
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_input_resenet50

resnet50_model = ResNet50(weights='imagenet', include_top=False)


def get_resnet50_features(x):
    n_samples = x.shape[0]
    x = K.cast_to_floatx(x)
    x = preprocess_input_resenet50(x)
    x = resnet50_model.predict(x)
    x = x.reshape((n_samples, -1))
    return x
