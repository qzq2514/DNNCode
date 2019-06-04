#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import json
import cv2
import os
from nets import MobileNetForDigit
import time

# is_training不使用placeholder的情况下,则在测试时,为了将is_training置为False,
# 只能重新手动加载模型,将is_training=False(见99行)
# 此方法对ckpt模型有效



model_path = 'models/ckpt/MobileNetForDigit3C_M2.ckpt-2000'
eval_dir="D:/forTensorflow/charRecTrain/test/"
label_map_path= "label_map.json"

support_image_extensions=[".jpg",".png",".jpeg",".bmp"]

channals=3
img_width=28
img_height=28

def create_label_map():
    label_dict={}
    label_id=0
    labels=[]
    for label in os.listdir(eval_dir):
        labels.append(label)
    labels.sort()
    for label in labels:
        label_dict[label]=label_id
        label_id+=1
    print(label_dict)
    with open(label_map_path,"w") as fw:
        label_map_str=json.dumps(label_dict,indent=4)
        fw.write(label_map_str)
    return label_dict

def load_label_map(replace=False):
    #创建新label_map
    if not os.path.exists(label_map_path) or replace:
        return create_label_map()

    #使用label_map
    with open(label_map_path,"r") as fr:
        # label_dict=eval(fr.read())
        label_dict=json.load(fr)
    return label_dict

label_dict=load_label_map(True)

def get_images_path(image_dir):
    image_paths=[]
    labels=[]

    for label_str in os.listdir(image_dir):
        class_dir=os.path.join(image_dir,label_str)
        for file_name in os.listdir(class_dir):
            extension=os.path.splitext(file_name)[1].lower()
            if extension not in support_image_extensions:
                continue
            image_path=os.path.join(image_dir,label_str,file_name)
            image_paths.append(image_path)
            # print(image_path)
            labels.append(label_dict[label_str])

    data_path_array=np.array([image_paths,labels])   #2xN大小,其中N是总样本数
    data_path_array=data_path_array.transpose()

    np.random.shuffle(data_path_array)
    image_paths=list(data_path_array[:,0])
    labels=data_path_array[:,1]

    #image_paths和labels都是列向量形式
    return np.array(image_paths),np.array(labels)


def get_data(image_path):
    channal_flag = cv2.IMREAD_GRAYSCALE if channals == 1 else cv2.IMREAD_COLOR
    image = cv2.imread(image_path, channal_flag)
    image = cv2.resize(image, (img_width, img_height))
    # image = image - 128.0                #在训练过程中已经进行了0均值化和归一化,这里不要再进行一次
    # image = 1.0 * image / 128.0
    image = np.resize(image, (img_height, img_width, channals))
    image = np.array(image, dtype=np.float32)
    return image

input_width=28
input_height=28

def eval():

    input_placeholder = tf.placeholder(tf.float32, shape=[None, input_width, input_height, channals], name="inputs")
    labels_placeholder = tf.placeholder(tf.int32, shape=[None], name="labels")

    classModel = MobileNetForDigit.MobileNetForDigit(is_training=False, num_classes=len(label_dict))
    preprocessed_inputs = classModel.preprocess(input_placeholder)
    logits = classModel.inference(preprocessed_inputs)

    softmax_output, classes = classModel.postprocess(logits)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        images_paths, labels = get_images_path(eval_dir)

        correct_cnt=0
        total_num=0
        for image_path in images_paths:
            image = cv2.imread(image_path)
            image_input=get_data(image_path)

            image_data=image_input[np.newaxis,:]

            s_time = time.time()
            softmax_out,pred_class=sess.run([softmax_output,classes],
                                            feed_dict={input_placeholder:image_data})
            usedTime = time.time() - s_time

            label_id=pred_class[0]
            isCorrect=label_id==int(labels[total_num])
            correct_cnt=correct_cnt+1 if isCorrect else correct_cnt
            total_num+=1
            labe_str=[key for key,value in label_dict.items() if value== label_id]

            print("accuary:{},usedTime:{}".format(correct_cnt/total_num,usedTime))
            print("labe_str:",labe_str[0])
            # if not isCorrect:
            #     cv2.imshow("image",image)
            #     cv2.waitKey(0)
eval()
