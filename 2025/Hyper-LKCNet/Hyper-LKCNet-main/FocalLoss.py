from keras import backend as BK
import tensorflow as tf
from keras.losses import Loss

class FocalLoss(Loss):
    def __init__(self, gamma=2, weight=None,label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        eps = BK.epsilon()
        # Apply label smoothing
        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true = (1 - self.label_smoothing) * y_true + self.label_smoothing / num_classes

        y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], tf.shape(y_pred)[1], -1))
        target = tf.reshape(y_true, tf.shape(y_pred))

        ce = -1 * tf.math.log(y_pred + eps) * target
        floss = tf.pow((1 - y_pred), self.gamma) * ce

        if self.weight is not None:
            floss = floss * self.weight

        floss = tf.reduce_sum(floss, axis=1)
        return tf.reduce_mean(floss)
