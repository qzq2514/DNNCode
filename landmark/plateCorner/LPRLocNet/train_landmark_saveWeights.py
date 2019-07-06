import cv2
import os

from keras.models import Model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D, Input, AveragePooling2D
import keras

from nets import LprLocNet

import numpy as np

# images_dir = "D:/forTensorflow/plateLandmarkDetTrain2/demo/images"
images_dir = "D:/forTensorflow/plateLandmarkDetTrain2/TX/images"
# image_dir="D:/forTensorflow/plateLandmarkDetTrain/All/test/images/"
# image_dir="D:/forTensorflow/plateLandmarkDetTrain/WI/images/"

save_model_name="models/plateCornerDet"

input_width=80
input_height=40
channals_num=3
batch_size=64

image_ext="jpg"
label_ext="json"

def loadData(image_dir,landmark_object):
    imgs_path = os.listdir(image_dir)

    np.array(imgs_path)

    images_data=[]
    labels=[]
    for img_path in imgs_path:
        img_path = os.path.join(image_dir, img_path)

        image = cv2.imread(img_path)
        (org_h, org_w) = image.shape[:2]

        input_image=landmark_object.preprocess(image)
        images_data.append(input_image)

        if len(images_data)%5==0:
            print("Successfully load {} pictures.".format(len(images_data)))


        # 加载标签
        batch_img_path=img_path.replace("images","labels").replace(image_ext,label_ext)

        cur_label=[]
        with open(batch_img_path,"r") as fr:
            postions=eval(fr.read())
            # postions = list(map(int, fr.readline().strip().split(" ")))
            normalize_postions = []
            for ind, pos in enumerate(postions):
                if ind % 2 == 0:
                    normalize_postions.append(pos / org_w)
                    # normalize_postions.append(pos / w * image_width)
                else:
                    normalize_postions.append(pos / org_h)
                    # normalize_postions.append(pos / h * image_height)
            cur_label.extend(normalize_postions)
            labels.append(cur_label)

    return np.array(images_data),np.array(labels)

def train():
    landmark_object = LprLocNet.LprLocNet(input_width, input_height, channals_num, is_training=True)
    landmark_model = landmark_object.constructDetModel()
    print(landmark_model.summary())

    print("Start training...")
    training_data,training_label= loadData(images_dir,landmark_object)
    print("Successflly load {} pictures and labels...".format(len(training_data)))

    # sgd = keras.optimizers.SGD(lr=0.05, decay=1e-3, momentum=0.9, nesterov=True)
    # model.compile(loss='mae', optimizer=sgd,metrics = ['mae'])
    # # model.compile(optimizer='rmsprop', loss='mse')
    #
    # for ind in range(100):
    #     model.fit(training_data,training_label,validation_split = 0.2, epochs = 100, batch_size = 64, verbose = 1)
    #     model.save("models/plateCornerDet-"+str(ind)+".h5")

    lr = 0.002
    decay = 0.9
    for i in range(100):
        print("lr:", lr)
        adam = keras.optimizers.Adam(lr=lr)
        landmark_model.compile(loss='mae', optimizer='adam', metrics=['mae'])
        train_history = landmark_model.fit(x=training_data, y=training_label, validation_split=0.2,
                                           epochs=10, batch_size=batch_size, verbose=1)
        lr *= decay
        if i %10==0:
            # 仅仅保存模型参数,下次预测的时候不仅需要从h5文件中读取权重参数(load_weights),
            # 还需要重新加载模型
            landmark_model.save_weights(save_model_name+str(i)+".h5")
            print("Succcessfully save model:", save_model_name + str(i) + ".h5")

train()