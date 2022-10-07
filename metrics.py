import keras.backend as K
import numpy as np
import tensorflow as tf




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





