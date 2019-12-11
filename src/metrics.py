
from keras import backend as K

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_postives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_postives + K.epsilon())


def f1(y_true, y_pred):

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    return 2* ((p * r)/ (p + r + K.epsilon()))

