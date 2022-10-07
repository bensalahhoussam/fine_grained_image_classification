import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.layers import Input, Reshape, Lambda, Dense, Activation
from keras.models import Sequential, Model
from keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from fine_grained_data_preparation import load_data
from keras.preprocessing.image import ImageDataGenerator
from metrics import f1_score
from loss_function import ategorical_focal_loss_with_label_smoothing
x_train, y_train = load_data("train")
x_valid, y_valid = load_data("valid")



def data(x_train,y_train):
    image=tf.image.resize(x_train,(224,224)
    image=tf.cast(image,tf.float32)/255.
    y_train_1= K.one_hot(y_train, num_classes=5)
    return image,y_train_1
def train_generator(x,y):
  train=tf.data.Dataset.from_tensor_slices((x,y)).map(data).shuffle(1000).batch(32).prefetch(4)
  return train
def valid_generator(x,y):
    train = tf.data.Dataset.from_tensor_slices((x, y)).map(data).shuffle(1000).batch(32)
    return train



train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest',
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train = train_datagen.flow_from_directory('D://Deep_Learning_projects/new_projects/computer_vision/skin_dataset/train/',
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=32,
        subset='training',
        class_mode='categorical')
val = test_datagen.flow_from_directory('D://Deep_Learning_projects/new_projects/computer_vision/skin_dataset/valid/',
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=32,
        subset='training',
        class_mode='categorical')


def dot_product(x):
    return K.batch_dot(x[0], x[1], axes=[1, 1]) / x[0].get_shape().as_list()[1]
def signed_sqrt(x):
    return K.sign(x) * K.sqrt(K.abs(x) + 1e-9)
def L2_norm(x, axis=-1):
    return K.l2_normalize(x, axis=axis)
def build_model():
    tensor_input = Input(shape=[224, 224, 3])
    model_detector_1 = ResNet50(input_tensor=tensor_input, include_top=False, weights='imagenet')
    model_detector_2 = ResNet50(input_tensor=tensor_input, include_top=False, weights='imagenet')
    model_detector_2 = Sequential(layers=model_detector_2.layers)
    for i, layer in enumerate(model_detector_2.layers):
        layer._name = layer.name + "_second"
    model2 = Model(inputs=[tensor_input], outputs=[model_detector_2.output])

    x = model_detector_1.layers[-1].output
    z = model_detector_1.layers[-1].output_shape
    y = model2.layers[-1].output
    #   rehape to (batch_size, total_pixels, filter_size)

    x = Reshape([z[1] * z[2], z[-1]])(x)
    y = Reshape([z[1] * z[2], z[-1]])(y)
    #outer products of x, y
    x = Lambda(dot_product)([x, y])
    x = Reshape([z[-1] * z[-1]])(x)
    # signed_sqrt
    x = Lambda(signed_sqrt)(x)
    #   L2_norm
    x = Lambda(L2_norm)(x)
    # FC - Layer
    x = Dense(units=5, kernel_initializer="he_normal")(x)
    prediction = Activation("softmax")(x)
    model_bilinear = Model(inputs=[tensor_input], outputs=[prediction])
    for layer in model_detector_1.layers:
        layer.trainable = False
    model_bilinear.compile(loss=categorical_focal_loss_with_label_smoothing(alpha=0.25, factor=0.1, gamma=2),
                           optimizer=Adam(),
                           metrics=["categorical_accuracy"])

    return model_bilinear

