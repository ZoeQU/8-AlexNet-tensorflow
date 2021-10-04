# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import Model
from keras.applications.imagenet_utils import (decode_predictions, preprocess_input)
from keras.layers import (Conv2D, Dense, Flatten, Dropout, Input, MaxPooling2D)
from keras.preprocessing import image
from keras.utils import plot_model

# from keras.datasets import cifar10
#
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()


def AlexNet(input_shape=None, num_classes=2):
    image_input = Input(shape=(224, 224, 3))
    # 224,224,3 -> 55,55,96 -> 27,27,96
    x = Conv2D(96,(11,11),strides=(4,4),activation='relu', padding='valid', kernel_initializer='uniform', name='conv1')(image_input)
    x = MaxPooling2D((3,3),strides = (2,2),name = 'pool1')(x)

    # 27,27,96 -> 27,27,256 -> 13,13,256
    x = Conv2D(256,(5,5),strides=(1,1), activation='relu',padding='same',name='conv2')(x)
    x = MaxPooling2D((3,3),strides = (2,2),name = 'pool2')(x)

    # 13,13,256 -> 13,13,384
    x = Conv2D(384,(3,3),strides=(1,1),activation='relu',padding='same',name='conv3')(x)

    # 13,13,384 -> 13,13,384
    x = Conv2D(384,(3,3),strides=(1,1),activation='relu',padding='same',name='conv4')(x)

    # 13,13,384 -> 13,13,256
    x = Conv2D(256,(3,3),strides=(1,1),activation='relu',padding='same',name='conv5')(x)
    x = MaxPooling2D((3,3),strides = (2,2),name = 'pool3')(x)

    # 7,7,512 -> 25088 -> 4096 -4096 -> num_classes
    x = Flatten(name = 'flatten')(x)
    x = Dense(4096,activation='relu',name='fc1')(x)
    x = Dropout(0.5,name='Drop1')(x)
    x = Dense(4096,activation='relu',name='fc2')(x)
    x = Dropout(0.5,name='Drop2')(x)
    x = Dense(num_classes, activation='softmax',name='predictions')(x)

    model = Model(image_input, x, name='AlexNet')

    return model


# if __name__ == '__main__':
#     model = AlexNet(2)
#     model.summary()
#     plot_model(model, "alexnet.svg", show_shapes=True)