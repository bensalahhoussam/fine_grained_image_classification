import keras.backend as K
import numpy as np
import tensorflow as tf
import keras.backend as K








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


