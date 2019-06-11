from keras.models import Model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D, Input, AveragePooling2D
import keras
import numpy as np
import cv2


class LprLocNet(object):
    def __init__(self,input_width,input_height,channals_num,is_training):
        self.is_training = is_training
        self.input_width = input_width
        self.input_height = input_height
        self.channals_num = channals_num

    def constructDetModel(self):
        input_shape=(self.input_height,self.input_width,self.channals_num)

        inputs = Input(shape=input_shape)

        x1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(inputs)
        x11 = Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding="same", activation="relu")(x1)
        x2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(4, 4), padding="same", activation="relu")(inputs)
        x3 = Conv2D(filters = 32, kernel_size = (5, 5), strides = (4, 4), padding = "same", activation = "relu")(x1)

        # x = keras.layers.concatenate([x1, x2], axis=-1)
        x112 = keras.layers.add([x11, x2])
        x112_pool = AveragePooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same")(x112)
        net = keras.layers.add([x112_pool, x3])

        if self.is_training:
            net = Dropout(0.5)(net)

        net = Flatten()(net)
        net = Dense(128, activation="tanh")(net)
        net = Dense(8, activation="tanh")(net)
        model = Model(inputs, net)

        return model

    def preprocess(self,image_org):

        image = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
        if self.channals_num == 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image_norm = image / 255
        image_resized = cv2.resize(image_norm, (self.input_width, self.input_height))
        image_reshape_np = np.resize(image_resized, (self.input_height,
                                                     self.input_width, self.channals_num))
        return image_reshape_np
