import keras.backend as K
import numpy as np
import tensorflow as tf
import keras.backend as K





def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        epsilon = K.epsilon()

        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1 - y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss

    return focal_loss

def categorical_label_smoothing(factor):
    def label_smoothing(y_actual,y_output):
        y_true_smooth = (1 - factor) * y_actual + (factor / 5.)
        categorical_crossentropy = -1 * y_true_smooth * K.log(y_output)
        return categorical_crossentropy 
    return label_smoothing


def categorical_focal_loss_with_label_smoothing(alpha, factor, gamma):
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


