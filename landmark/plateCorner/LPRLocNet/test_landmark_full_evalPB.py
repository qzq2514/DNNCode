#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from nets import LprLocNet
import json
import time
import cv2
import os


model_path = "models/plateCornerDetFull30.pb"
eval_dir=   "D:/forTensorflow/plateLandmarkDetTrain2/TX/images"
support_image_extensions=[".jpg",".png",".jpeg",".bmp"]

input_width = 80
input_height = 40
channals_num=3

show_width = 400
show_height = 200

def eval():
    lanmark_object = LprLocNet.LprLocNet(input_width, input_height, channals_num, is_training=True)

    #加载pb模型
    with tf.Session() as sess:
        with tf.gfile.FastGFile(model_path,"rb") as fr:
            graph_def=tf.GraphDef()
            graph_def.ParseFromString(fr.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def,name="")

        sess.run(tf.global_variables_initializer())

        _landmark_output = sess.graph.get_tensor_by_name('landmark_output:0')
        _inputs = sess.graph.get_tensor_by_name('input_1:0')


        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (150, 150, 150)]
        for j, image_file in enumerate(os.listdir(eval_dir)):
            if not image_file.endswith("jpg"):
                continue
            image_ori = cv2.imread(os.path.join(eval_dir, image_file))
            org_height, org_width = image_ori.shape[:2]

            processed_image = lanmark_object.preprocess(image_ori)
            image_data = np.array([processed_image])
            time1 = time.time()
            landmark_output = sess.run(_landmark_output, feed_dict={_inputs: image_data})
            print("time:", time.time() - time1)

            pst1 = []
            # print(landmark_output)
            for i in range(4):
                print(landmark_output[0][i], landmark_output[0][i + 1])
                x = int(landmark_output[0][2 * i] * org_width)
                y = int(landmark_output[0][2 * i + 1] * org_height)
                pst1.append([x, y])
                cv2.circle(image_ori, (x, y), 2, colors[i], 2)
            pst1 = np.float32(pst1)

            past2 = np.float32([[0, 0], [show_width, 0], [show_width, show_height], [0, show_height]])
            M = cv2.getPerspectiveTransform(pst1, past2)
            dst = cv2.warpPerspective(image_ori, M, (show_width, show_height))
            cv2.imshow("img", image_ori)
            cv2.imshow("img2", dst)
            cv2.waitKey(0)

if __name__=="__main__":
    eval()
