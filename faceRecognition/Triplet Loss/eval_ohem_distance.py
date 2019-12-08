import numpy as np
import tensorflow as tf
import json
import time
import cv2
import os
import dataset

model_path = 'models/pb/siamese_triplet_28out_bn_face2-200.pb'
eval_root =   "../dataBase40"

batch_size=16
input_height=32
input_width=32
channals=3
support_image_extensions=[".jpg",".png",".jpeg",".bmp"]

sess=tf.Session()
with tf.gfile.FastGFile(model_path, "rb") as fr:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fr.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name="")

sess.run(tf.global_variables_initializer())
input = sess.graph.get_tensor_by_name('anchor:0')
output = sess.graph.get_tensor_by_name('flatten_anchor:0')

def get_dist(feature1,feature2):
    return np.mean((feature1 - feature2) ** 2)

def eval(test_dataset):
        while True:
            anchor_path, similar_path, dissimilar_path,labels = \
                test_dataset.get_next_triplet_paths()

            anchor = test_dataset.process_data(anchor_path)[np.newaxis,:]
            pos = test_dataset.process_data(similar_path)[np.newaxis,:]
            neg = test_dataset.process_data(dissimilar_path)[np.newaxis,:]

            start_time=time.time()
            anchor=sess.run(output,feed_dict={input:anchor})
            print("time:{}".format(time.time() - start_time))

            pos_output = sess.run(output, feed_dict={input: pos})
            neg_output = sess.run(output, feed_dict={input: neg})

            pos_dist=get_dist(anchor,pos_output)
            neg_dist=get_dist(anchor,neg_output)

            print("pos_dist:",pos_dist)
            print("neg_dist:",neg_dist)
            print("-------------------------------------------")

            anchor_show = cv2.imread(anchor_path)
            similar_show = cv2.imread(similar_path)
            dissimilar_show = cv2.imread(dissimilar_path)

            cv2.imshow("anchor", anchor_show)
            cv2.imshow("similar", similar_show)
            cv2.imshow("dissimilar", dissimilar_show)

            cv2.imwrite("anchor.jpg",anchor_show)
            cv2.imwrite("similar.jpg", similar_show)
            cv2.imwrite("dissimilar.jpg", dissimilar_show)
            cv2.waitKey(0)


if __name__=="__main__":
    test_dataset = dataset.dataset(eval_root, batch_size, support_image_extensions,
                                   input_height, input_width, channals)
    eval(test_dataset)
