from keras.layers import Concatenate,ReLU,Lambda,AveragePooling2D,Permute,Add
from keras import backend as BK
from keras.layers import Layer
import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import Activation,GlobalAveragePooling2D

class MultiScaleFeatureExtractor(Layer):
    def __init__(self, channel, **kwargs):
        super(MultiScaleFeatureExtractor, self).__init__(**kwargs)
        self.conv1 = Conv2D(channel, (1, 1), strides=(1, 1), padding='same', use_bias=False)
        self.conv3 = Conv2D(channel, (3, 3), strides=(1, 1), padding='same', use_bias=False)
        self.conv5 = Conv2D(channel, (5, 5), strides=(1, 1), padding='same', use_bias=False)

    def call(self, x):
        f1 = self.conv1(x)
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        fused_features = Add()([f1, f3, f5])
        return fused_features

class CA_Block(Layer):
    def __init__(self, channel, h, w, kernel_size=1, **kwargs):
        super(CA_Block, self).__init__(**kwargs)
        self.channel = channel
        self.kernel_size = kernel_size
        self.h = h
        self.w = w

        self.avg_pool_x = AveragePooling2D(pool_size=(h, 1), data_format='channels_last')
        self.avg_pool_y = AveragePooling2D(pool_size=(1, w), data_format='channels_last')

        self.conv_1x1 = Conv2D(filters=channel, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)
        self.relu = ReLU()
        self.multi_scale_extractor = MultiScaleFeatureExtractor(channel)

        self.conv_h = Conv2D(filters=channel, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', use_bias=False)
        self.conv_w = Conv2D(filters=channel, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', use_bias=False)

        self.sigmoid_h = Activation('sigmoid')
        self.sigmoid_w = Activation('sigmoid')

    def call(self, x):
        x_h = self.avg_pool_x(x)
        x_w = self.avg_pool_y(x)
        x_h = Lambda(lambda t: BK.expand_dims(t, axis=-1))(x_h)
        x_w = Lambda(lambda t: BK.expand_dims(t, axis=-1))(x_w)
        # 移除维度大小为 1 的维度
        x_h = Lambda(lambda t: BK.squeeze(t, axis=-1))(x_h)
        x_w = Lambda(lambda t: BK.squeeze(t, axis=-1))(x_w)
        x_h_permuted = Permute((2, 1, 3))(x_h)

        x_cat_conv_relu = self.relu(self.conv_1x1(Concatenate(axis=-3)([x_h_permuted, x_w])))
        print(x_cat_conv_relu.shape)
        x_cat_conv_relu = self.multi_scale_extractor(x_cat_conv_relu)
        x_cat_conv_relu = self.relu(x_cat_conv_relu)

        # Modify this line in your call method
        x_cat_conv_split_h, x_cat_conv_split_w = Lambda(
            lambda x: tf.split(x, [self.h, self.w], axis=-3))(x_cat_conv_relu)
        print(x_cat_conv_split_h.shape)
        print(x_cat_conv_split_w.shape)
        x_cat_conv_split_h=Permute((2,1,3))(x_cat_conv_split_h)
        s_h = self.sigmoid_h(self.conv_h(x_cat_conv_split_h))
        s_w = self.sigmoid_w(self.conv_w(x_cat_conv_split_w))

        out = s_h * s_w

        return out

class ChannelAttention(Layer):
    def __init__(self, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv_1 = Conv2D(1, (1, 1), strides=(1, 1), padding='same', use_bias=False)
        self.relu = Activation('relu')
        self.conv_channel = Conv2D(input_shape[-1], (1, 1), strides=(1, 1), padding='same', use_bias=False)
        self.sigmoid = Activation('sigmoid')
        super(ChannelAttention, self).build(input_shape)

    def call(self, x):
        x_conv = self.conv_1(x)
        x_conv_relu = self.relu(x_conv)

        # Global Average Pooling on channel dimension
        x_pool = GlobalAveragePooling2D()(x_conv_relu)
        x_pool = BK.expand_dims(x_pool, axis=1)
        x_pool = BK.expand_dims(x_pool, axis=1)

        x_attention = self.conv_channel(x_pool)
        x_attention = self.sigmoid(x_attention)

        out = x_attention

        return out

class CCA(Layer):
    def __init__(self,**kwargs):
        super(CCA, self).__init__(**kwargs)

    def build(self, input_shape):
        _, h, w, c = input_shape
        self.channel_att= ChannelAttention()
        self.spatial_att = CA_Block(channel=c, h=h, w=w)
        self.sigmoid = Activation('sigmoid')

    def call(self, x):
        att = 1 + self.sigmoid(self.channel_att(x)*self.spatial_att(x))
        return att * x