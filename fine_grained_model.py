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

"""x_train, y_train = load_data("train")
x_valid, y_valid = load_data("valid")"""
"""print(x_train.shape, x_valid.shape)
print(y_train.shape, y_valid.shape)"""

def data(x_train,y_train):
    image=tf.image.resize(x_train,(150,150))
    image=tf.cast(image,tf.float32)/255.
    y_train_1= K.one_hot(y_train, num_classes=5)
    return image,y_train_1
def train_generator(x,y):
  train=tf.data.Dataset.from_tensor_slices((x,y)).map(data).shuffle(1000).batch(32).prefetch(4)
  return train
def valid_generator(x,y):
    train = tf.data.Dataset.from_tensor_slices((x, y)).map(data).shuffle(1000).batch(32)
    return train
def categorical_focal_loss_with_label_smoothing(alpha, factor,gamma):
    def focal_loss(y_actual, y_output):
        epsilion = K.epsilon()
        y_output_1 = K.clip(y_output, epsilion, 1.0 - epsilion)
        y_true_smooth = (1 - factor) * y_actual + (factor / 5.)
        cross_entropy = -1 * y_true_smooth * K.log(y_output_1)
        weight = alpha * y_true_smooth * np.power((1 - y_output_1), gamma)
        loss = cross_entropy * weight
        loss = K.sum(loss, axis=1)
        return loss
    return focal_loss
"""y_pred=K.softmax(np.abs(np.round(np.random.normal(size=(10,2)),2)),axis=1)
print(y_pred)
y_pred_1=K.one_hot(K.argmax(y_pred,axis=1),num_classes=2)
print("y_pred",y_pred_1)
y_1=np.tile(np.array([[1.0,0.0]]),(5,1))
y_2=np.tile(np.array([[0.0,1.0]]),(5,1))

y_pred_2 = tf.convert_to_tensor(np.array([[1., 0.],
                         [1., 0.],
                         [1., 0.],
                         [1., 0.],
                         [0., 1.],
                         [1., 0.],
                         [0., 1.],
                         [0., 1.],
                         [0., 1.],
                         [0., 1.]]))
y_actual=np.concatenate([y_1,y_2],axis=0)
print("y_actual",tf.convert_to_tensor(y_actual))
print("*"*50)

print(y_actual*y_pred_2)
true_positives=K.sum(y_actual*y_pred_2)
possible_positives=K.sum(y_actual)
recal=(true_positives/possible_positives)*100
print(true_positives)
print(possible_positives)
print(recal)"""
def f1_score(y_actual, y_output):
    def recall(y_actual, y_output):
        y_pred_1 = K.one_hot(K.argmax(y_output, axis=1), num_classes=5)
        true_positives = K.sum(y_actual * y_pred_1)
        possible_positives = K.sum(y_actual)
        recall = true_positives / possible_positives
        return recall
    def precision(y_actual, y_output):
        y_pred_1 = K.one_hot(K.argmax(y_output, axis=1), num_classes=5)
        true_positives = K.sum(y_actual * y_pred_1)
        predicted_positives = K.sum(K.round(K.clip(y_pred_1, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_actual, y_output)
    recall = recall(y_actual, y_output)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


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
train = train_datagen.flow_from_directory(
        'D://Deep_Learning_projects/new_projects/computer_vision/skin_dataset/train/',
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=32,
        subset='training',
        class_mode='categorical')
val = test_datagen.flow_from_directory(
        'D://Deep_Learning_projects/new_projects/computer_vision/skin_dataset/valid/',
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

    x = Reshape([z[1] * z[2], z[-1]])(x)
    y = Reshape([z[1] * z[2], z[-1]])(y)
    x = Lambda(dot_product)([x, y])
    x = Reshape([z[-1] * z[-1]])(x)
    x = Lambda(signed_sqrt)(x)
    x = Lambda(L2_norm)(x)
    x = Dense(units=5, kernel_initializer="he_normal")(x)
    prediction = Activation("softmax")(x)
    model_bilinear = Model(inputs=[tensor_input], outputs=[prediction])
    for layer in model_detector_1.layers:
        layer.trainable = False
    model_bilinear.compile(loss=categorical_focal_loss_with_label_smoothing(alpha=0.25, factor=0.1, gamma=2),
                           optimizer=Adam(),
                           metrics=["categorical_accuracy"])

    return model_bilinear
#model = build_model()
#.fit(x_train,y_train,batch_size=32,steps_per_epoch=len(x_train)//32,epochs=20,validation_data=(x_test,y_test),validation_steps=len(x_test)//32)
model1= ResNet50(input_shape=(224,224,3), include_top=False, weights='imagenet')
print(model1.summary())
z = model1.layers[-1].output_shape

x=model1.layers[-1].output
x=Reshape([z[1] * z[2], z[-1]])(x)
print(x.shape)