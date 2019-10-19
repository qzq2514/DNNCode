import numpy as np
import tensorflow as tf
import json
import time
import cv2
import os
import dataset

model_path = 'models/pb/siamese_face_bn-300.pb'
eval_root=   "P:/WorkSpace/VS2015/FaceRecognition/FaceRecognition/trainData/faceData/dataBase41"

batch_size=16
input_height=32
input_width=32
channals=3
support_image_extensions=[".jpg",".png",".jpeg",".bmp"]

def eval(test_dataset):
    with tf.Session() as sess:
        with tf.gfile.FastGFile(model_path,"rb") as fr:
            graph_def=tf.GraphDef()
            graph_def.ParseFromString(fr.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def,name="")

        sess.run(tf.global_variables_initializer())

        input = sess.graph.get_tensor_by_name('input_left:0')
        output = sess.graph.get_tensor_by_name('flatten_out:0')

        correct_cnt=0
        total_num=0
        while True:
            image_left_path,image_right_path,pair_label=\
                test_dataset.get_one_pair_path()
            image_left_data=test_dataset.process_data(image_left_path)[np.newaxis,:]
            image_right_data = test_dataset.process_data(image_right_path)[np.newaxis,:]

            start_time=time.time()
            left_output=sess.run(output,feed_dict={input:image_left_data})
            right_output = sess.run(output, feed_dict={input: image_right_data})
            similarity = np.mean((left_output - right_output) ** 2)
            print("time:{}".format(time.time()-start_time))
            # print("left_output:",left_output)
            # print("right_output:",right_output)

            print("pair_label:{}----->distance:{}".format(pair_label,similarity))
            print("-------------------------------------------")

            image_left = cv2.imread(image_left_path)
            image_right = cv2.imread(image_right_path)
            cv2.imshow("left", image_left)
            cv2.imshow("right", image_right)
            cv2.waitKey(0)



if __name__=="__main__":
    test_dataset = dataset.dataset(eval_root, batch_size, support_image_extensions,
                                   input_height, input_width, channals)
    eval(test_dataset)